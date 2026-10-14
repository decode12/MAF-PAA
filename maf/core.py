"""MAF model, DGM4 preprocessing and objectives used by the final paper.

Derived from the frozen implementation SHA256
cd58e939e6a8022cb5f3a238300693e5a302d79a106568b93f20b84730fbb7e2.
Only final MAF and complete-module ablations are included. Dataset annotations
supply training targets and evaluation labels; predictions never use them.
"""
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel

CORE_ABLATIONS = ("full", "without_mgr_complete", "without_haf_complete", "without_dafo_complete")
ABLATIONS = CORE_ABLATIONS
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def loader_generator(seed, split):
    """Keep sampling/worker seeds independent of architecture RNG consumption."""
    offsets = {'train': 0, 'val': 1000000, 'test': 2000000}
    return torch.Generator().manual_seed(int(seed) + offsets[split])

def first_epoch_sampler_manifest(loader):
    """Preview the real shuffled order without changing its generator or reading data."""
    if loader.generator is None:
        raise ValueError('An explicit DataLoader generator is required')
    before = loader.generator.get_state().clone()
    copied = torch.Generator()
    copied.set_state(before)
    preview = DataLoader(torch.arange(len(loader.dataset)), batch_size=loader.batch_size, shuffle=True, generator=copied, num_workers=0, drop_last=loader.drop_last)
    order = torch.cat(list(preview)).to(torch.int64).numpy().astype('<i8', copy=False)
    assert torch.equal(before, loader.generator.get_state())
    order_sha256 = hashlib.sha256(order.tobytes()).hexdigest()
    return {'sample_count': int(order.size), 'n_records': int(order.size), 'index_encoding': 'little-endian int64', 'sha256': order_sha256, 'order_sha256': order_sha256, 'first_64_indices': order[:64].tolist(), 'generator_state_unchanged': True}

def module_state_sha256(module):
    """Digest named tensor metadata and exact initialized bytes, including buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def load_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    stripped = text.lstrip()
    if stripped.startswith('['):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def resolve_image_path(data_root: Path, rel: str):
    """
    Handles both:
      data_root=/.../datasets
      rel=DGM4/...
    and:
      data_root=/.../datasets/DGM4
      rel=DGM4/...
    """
    rel = str(rel).replace('\\', '/')
    candidates = [data_root / rel, data_root.parent / rel]
    if rel.startswith('DGM4/'):
        candidates.append(data_root / rel[len('DGM4/'):])
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def whitespace_word_spans(text: str):
    """
    Official DGM4 fake_text_pos uses zero-based whitespace-token indices.
    Example [8,13,17] -> celebrate, event, won.
    """
    spans = []
    for m in re.finditer('\\S+', text):
        spans.append((m.start(), m.end()))
    return spans

def text_token_targets(text, fake_word_indices, offsets, attention_mask):
    """
    Map DGM4 word indices to CLIP subword-token targets using char offsets.
    """
    max_len = len(offsets)
    target = torch.zeros(max_len, dtype=torch.float32)
    valid = torch.zeros(max_len, dtype=torch.float32)
    word_spans = whitespace_word_spans(text)
    fake_spans = []
    for idx in fake_word_indices or []:
        if 0 <= int(idx) < len(word_spans):
            fake_spans.append(word_spans[int(idx)])
    for j, ((s, e), attn) in enumerate(zip(offsets, attention_mask)):
        if int(attn) == 0 or e <= s:
            continue
        valid[j] = 1.0
        for ws, we in fake_spans:
            if max(s, ws) < min(e, we):
                target[j] = 1.0
                break
    return (target, valid)

def bbox_to_patch_mask(box, orig_w, orig_h, image_size=224, patch_size=14):
    """
    We resize every image directly to image_size x image_size.
    A patch is positive if its center falls inside the resized GT bbox.
    """
    grid = image_size // patch_size
    mask = torch.zeros(grid * grid, dtype=torch.float32)
    if box is None or len(box) != 4:
        return mask
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1 or orig_w <= 0 or (orig_h <= 0):
        return mask
    sx = image_size / float(orig_w)
    sy = image_size / float(orig_h)
    x1, x2 = (x1 * sx, x2 * sx)
    y1, y2 = (y1 * sy, y2 * sy)
    idx = 0
    for r in range(grid):
        cy = (r + 0.5) * patch_size
        for c in range(grid):
            cx = (c + 0.5) * patch_size
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                mask[idx] = 1.0
            idx += 1
    return mask

def infer_labels(item):
    fake_cls = str(item.get('fake_cls', '')).lower().strip()
    fake_text_pos = item.get('fake_text_pos', []) or []
    fake_image_box = item.get('fake_image_box', []) or []
    text_manip = 1 if len(fake_text_pos) > 0 or 'text_' in fake_cls or 'text' in fake_cls else 0
    image_manip = 1 if len(fake_image_box) == 4 and fake_image_box[2] > fake_image_box[0] and (fake_image_box[3] > fake_image_box[1]) or 'face_' in fake_cls or 'face' in fake_cls or ('image' in fake_cls) else 0
    real_names = {'orig', 'original', 'real', 'none', 'pristine', 'clean'}
    is_real_by_name = fake_cls in real_names or fake_cls == ''
    fake = 0 if is_real_by_name and text_manip == 0 and (image_manip == 0) else 1
    if text_manip and image_manip:
        state = 3
    elif text_manip:
        state = 1
    elif image_manip:
        state = 2
    else:
        state = 0
    return (fake, text_manip, image_manip, state)

class DGM4Dataset(Dataset):

    def __init__(self, json_path, data_root, tokenizer, max_length=77, image_size=224, patch_size=14, source='bbc', source_by_id=None, limit=0):
        rows = load_json_or_jsonl(json_path)
        if source_by_id is None:
            raise ValueError('An explicit VisualNews ID-to-source mapping is required')
        missing = [row['id'] for row in rows if str(row['id']) not in source_by_id]
        if missing:
            raise ValueError(f'Missing VisualNews source for IDs: {missing[:5]}')
        # Stable sample IDs refer to the input split metadata row, not source-local order.
        self.raw_indices = [i for i, row in enumerate(rows) if source_by_id[str(row['id'])] == source]
        self.items = [rows[i] for i in self.raw_indices]
        self.split = Path(json_path).stem
        self.full_count = len(self.items)
        if limit:
            self.items = self.items[:limit]
            self.raw_indices = self.raw_indices[:limit]
        if not self.items:
            raise ValueError(f'No {source} records in {json_path}')
        print(f'[data] {Path(json_path).name}: source={source}, full={self.full_count}, used={len(self.items)}', flush=True)
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_tf = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = str(item['text'])
        image_path = resolve_image_path(self.data_root, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'Failed to load image: {image_path}') from e
        orig_w, orig_h = image.size
        pixel_values = self.image_tf(image)
        enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_offsets_mapping=True, return_tensors=None)
        offsets = enc.pop('offset_mapping')
        input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)
        fake_text_pos = item.get('fake_text_pos', []) or []
        token_target, token_valid = text_token_targets(text, fake_text_pos, offsets, enc['attention_mask'])
        patch_target = bbox_to_patch_mask(item.get('fake_image_box', []) or [], orig_w, orig_h, image_size=self.image_size, patch_size=self.patch_size)
        fake, text_manip, image_manip, state = infer_labels(item)
        return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'token_target': token_target, 'token_valid': token_valid, 'patch_target': patch_target, 'label': torch.tensor(fake, dtype=torch.float32), 'text_manip': torch.tensor(text_manip, dtype=torch.float32), 'image_manip': torch.tensor(image_manip, dtype=torch.float32), 'state': torch.tensor(state, dtype=torch.long), 'id': str(item.get('id', idx))}

class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)

class MAF(nn.Module):
    """Final MAF and its three complete module ablations.

Checkpoint-compatible parameter names and RNG consumption are retained. The
five *_corr parameter groups are inactive compatibility parameters: they do
not contribute to the prediction or loss. Their original initialization and
three discarded forward calls, plus two discarded presence calls, preserve
the seeded training trajectory. They are not alternative model options.
Only full, without_mgr_complete, without_haf_complete and without_dafo_complete
are exposed; no historical correction-stage model can be selected.
"""

    def __init__(self, clip_name='openai/clip-vit-base-patch16', hidden_dim=256, dropout=0.1, ablation='full'):
        super().__init__()
        self.ablation = 'full'
        self.stage = 'base'
        self.isolate_grounding = True
        self.base_frozen = False
        self.clip = CLIPModel.from_pretrained(clip_name)
        vision_h = self.clip.config.vision_config.hidden_size
        text_h = self.clip.config.text_config.hidden_size
        proj_dim = self.clip.config.projection_dim
        self.text_local_proj = nn.Sequential(nn.LayerNorm(text_h), nn.Linear(text_h, hidden_dim))
        self.image_local_proj = nn.Sequential(nn.LayerNorm(vision_h), nn.Linear(vision_h, hidden_dim))
        base_in = proj_dim * 4
        self.base_fusion = nn.Sequential(nn.LayerNorm(base_in), nn.Linear(base_in, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.base_head = nn.Linear(hidden_dim, 1)
        self.text_ground_head = MLP(hidden_dim * 2, hidden_dim, 1, dropout)
        self.image_ground_head = MLP(hidden_dim * 2, hidden_dim, 1, dropout)
        self.text_presence_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.image_presence_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        heads = 8 if hidden_dim % 8 == 0 else 4
        self.text_from_image = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.image_from_text = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        # Inactive parameters retained only for checkpoint and initialization RNG compatibility.
        self.ground_corr = MLP(hidden_dim * 3, hidden_dim, 1, dropout)
        self.residual_corr = MLP(hidden_dim * 5, hidden_dim, 1, dropout)
        self.text_corr = MLP(hidden_dim * 3, hidden_dim, 1, dropout)
        self.image_corr = MLP(hidden_dim * 3, hidden_dim, 1, dropout)
        self.joint_corr = MLP(hidden_dim * 7, hidden_dim * 2, 1, dropout)
        self.state_router = MLP(hidden_dim * 5, hidden_dim * 2, 4, dropout)
        self.veracity_head = MLP(hidden_dim * 8, hidden_dim * 2, 1, dropout)
        self.configure_ablation(ablation)

    def configure_ablation(self, ablation):
        if ablation not in ABLATIONS:
            raise ValueError(f'Unknown ablation: {ablation}')
        self.ablation = ablation
        if ablation == 'without_mgr_complete':
            hidden = self.base_head.in_features
            for name in ('text_local_proj', 'image_local_proj', 'text_ground_head', 'image_ground_head', 'text_from_image', 'image_from_text', 'ground_corr', 'residual_corr', 'text_corr', 'image_corr', 'joint_corr', 'state_router', 'veracity_head'):
                delattr(self, name)
            self.global_state_head = MLP(hidden, hidden * 2, 4, 0.1)
            self.global_veracity_head = MLP(hidden, hidden * 2, 1, 0.1)
        elif ablation == 'without_haf_complete':
            del self.text_presence_head
            del self.image_presence_head

    def global_replacement_forward(self, z0, base_logit):
        state_logits = self.global_state_head(z0)
        q = F.softmax(state_logits, dim=-1)
        evidence_logit = self.global_veracity_head(z0).squeeze(-1)
        pt = self.text_presence_head(z0).squeeze(-1)
        pv = self.image_presence_head(z0).squeeze(-1)
        state_fake = (1 - q[:, 0]).clamp(0.0001, 1 - 0.0001)
        state_logit = torch.log(state_fake) - torch.log1p(-state_fake)
        agreement = (1 - 0.5 * (torch.abs(q[:, 1] + q[:, 3] - pt.sigmoid()) + torch.abs(q[:, 2] + q[:, 3] - pv.sigmoid()))).detach()
        gate = (agreement * q.max(dim=-1).values.detach()).clamp(0, 1)
        ws = 0.08 + 0.07 * gate
        final = 0.35 * base_logit + (0.65 - ws) * evidence_logit + ws * state_logit
        return dict(base_logit=base_logit, final_logit=final, evidence_logit=evidence_logit, state_logits=state_logits, state_prob=q, state_fake_logit=state_logit, text_presence_logit=pt, image_presence_logit=pv, cross_head_agreement=agreement, fusion_gate=gate, state_coeff=ws)

    def set_clip_trainable(self, trainable: bool):
        for p in self.clip.parameters():
            p.requires_grad = trainable

    def train(self, mode=True):
        super().train(mode)
        if not any((p.requires_grad for p in self.clip.parameters())):
            self.clip.eval()
        if self.base_frozen:
            self.base_fusion.eval()
            self.base_head.eval()
        return self

    def _encode_clip(self, pixel_values, input_ids, attention_mask):
        clip_trainable = any((p.requires_grad for p in self.clip.parameters()))
        ctx = torch.set_grad_enabled(torch.is_grad_enabled() and clip_trainable)
        with ctx:
            vision_out = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
            text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            v_pool = vision_out.pooler_output
            t_pool = text_out.pooler_output
            z_v = self.clip.visual_projection(v_pool)
            z_t = self.clip.text_projection(t_pool)
            v_tokens = vision_out.last_hidden_state[:, 1:, :]
            t_tokens = text_out.last_hidden_state
        return (z_v, z_t, v_tokens, t_tokens)

    @staticmethod
    def weighted_pool(features, weights, valid=None, eps=1e-06):
        w = weights
        if valid is not None:
            w = w * valid
        denom = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return (features * w.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, batch):
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        z_v, z_t, v_raw, t_raw = self._encode_clip(pixel_values, input_ids, attention_mask)
        z_vn = F.normalize(z_v, dim=-1)
        z_tn = F.normalize(z_t, dim=-1)
        base_feat = torch.cat([z_vn, z_tn, torch.abs(z_vn - z_tn), z_vn * z_tn], dim=-1)
        z0 = self.base_fusion(base_feat)
        base_logit = self.base_head(z0).squeeze(-1)
        if self.ablation == 'without_mgr_complete':
            return self.global_replacement_forward(z0, base_logit)
        t = self.text_local_proj(t_raw)
        v = self.image_local_proj(v_raw)
        text_key_padding = ~attention_mask.bool()
        z0_t = z0.unsqueeze(1).expand(-1, t.size(1), -1)
        z0_v = z0.unsqueeze(1).expand(-1, v.size(1), -1)
        text_ground_logit = self.text_ground_head(torch.cat([t, z0_t], dim=-1)).squeeze(-1)
        image_ground_logit = self.image_ground_head(torch.cat([v, z0_v], dim=-1)).squeeze(-1)
        # These first presence calls preserve RNG consumption; the local-context calls below define the prediction.
        if self.ablation != 'without_haf_complete':
            text_presence_logit = self.text_presence_head(z0).squeeze(-1)
            image_presence_logit = self.image_presence_head(z0).squeeze(-1)
            p_text = torch.sigmoid(text_presence_logit)
            p_image = torch.sigmoid(image_presence_logit)
        wt = torch.sigmoid(text_ground_logit)
        wi = torch.sigmoid(image_ground_logit)
        if self.isolate_grounding:
            wt_corr = wt.detach()
            wi_corr = wi.detach()
        else:
            wt_corr = wt
            wi_corr = wi
        valid_text = batch.get('token_valid', attention_mask.float()).float()
        f_t = self.weighted_pool(t, wt_corr, valid_text)
        f_v = self.weighted_pool(v, wi_corr, None)
        t_pred, _ = self.text_from_image(query=t, key=v, value=v, need_weights=False)
        v_pred, _ = self.image_from_text(query=v, key=t, value=t, key_padding_mask=text_key_padding, need_weights=False)
        t_res_tokens = t - t_pred
        v_res_tokens = v - v_pred
        r_t = self.weighted_pool(t_res_tokens, wt_corr, valid_text)
        r_v = self.weighted_pool(v_res_tokens, wi_corr, None)
        # Deliberately discarded outputs preserve the original training Dropout RNG sequence.
        d_t_raw = self.text_corr(torch.cat([z0, f_t, r_t], dim=-1)).squeeze(-1)
        d_v_raw = self.image_corr(torch.cat([z0, f_v, r_v], dim=-1)).squeeze(-1)
        joint_feat = torch.cat([z0, f_t, f_v, r_t, r_v, r_t * r_v, torch.abs(r_t - r_v)], dim=-1)
        d_j_raw = self.joint_corr(joint_feat).squeeze(-1)
        if self.ablation != 'without_haf_complete':
            text_presence_logit = self.text_presence_head(z0 + 0.5 * f_t + 0.5 * r_t).squeeze(-1)
            image_presence_logit = self.image_presence_head(z0 + 0.5 * f_v + 0.5 * r_v).squeeze(-1)
            p_text = torch.sigmoid(text_presence_logit)
            p_image = torch.sigmoid(image_presence_logit)
        state_feat = torch.cat([z0, f_t, f_v, r_t, r_v], dim=-1)
        state_logits = self.state_router(state_feat)
        state_prob = F.softmax(state_logits, dim=-1)
        state_context = state_prob[:, 0:1] * z0 + state_prob[:, 1:2] * f_t + state_prob[:, 2:3] * f_v + state_prob[:, 3:4] * 0.5 * (f_t + f_v)
        veracity_feat = torch.cat([z0, f_t, f_v, r_t, r_v, state_context, r_t * r_v, torch.abs(r_t - r_v)], dim=-1)
        evidence_logit = self.veracity_head(veracity_feat).squeeze(-1)
        if self.ablation == 'without_haf_complete':
            return dict(base_logit=base_logit, final_logit=evidence_logit, evidence_logit=evidence_logit, state_logits=state_logits, state_prob=state_prob, text_ground_logit=text_ground_logit, image_ground_logit=image_ground_logit)
        p_fake_state = torch.clamp(1.0 - state_prob[:, 0], min=0.0001, max=1.0 - 0.0001)
        state_fake_logit = torch.log(p_fake_state) - torch.log1p(-p_fake_state)
        state_text_prob = state_prob[:, 1] + state_prob[:, 3]
        state_image_prob = state_prob[:, 2] + state_prob[:, 3]
        text_agreement = 1.0 - torch.abs(state_text_prob - p_text)
        image_agreement = 1.0 - torch.abs(state_image_prob - p_image)
        cross_head_agreement = (0.5 * text_agreement + 0.5 * image_agreement).detach()
        router_confidence = state_prob.max(dim=-1).values.detach()
        fusion_gate = torch.clamp(cross_head_agreement * router_confidence, min=0.0, max=1.0)
        state_coeff = 0.08 + 0.07 * fusion_gate
        evidence_coeff = 0.65 - state_coeff
        final_logit = 0.35 * base_logit + evidence_coeff * evidence_logit + state_coeff * state_fake_logit
        return {'base_logit': base_logit, 'final_logit': final_logit, 'evidence_logit': evidence_logit, 'state_logits': state_logits, 'state_prob': state_prob, 'state_fake_logit': state_fake_logit, 'cross_head_agreement': cross_head_agreement, 'fusion_gate': fusion_gate, 'state_coeff': state_coeff, 'text_ground_logit': text_ground_logit, 'image_ground_logit': image_ground_logit, 'text_presence_logit': text_presence_logit, 'image_presence_logit': image_presence_logit, 'p_text': p_text, 'p_image': p_image, 'z0': z0, 'f_t': f_t, 'f_v': f_v, 'r_t': r_t, 'r_v': r_v}

def masked_bce_with_logits(logits, targets, valid=None, pos_weight=5.0):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weight = torch.ones_like(targets)
    weight = torch.where(targets > 0.5, torch.full_like(weight, float(pos_weight)), weight)
    loss = loss * weight
    if valid is not None:
        loss = loss * valid
        denom = valid.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(targets.numel(), device=targets.device, dtype=targets.dtype)
    return loss.sum() / denom

def compute_loss(outputs, batch, args):
    """Return the exact joint objective and scalar diagnostics for a physical minibatch.

    args needs ablation, text_pos_weight (5) and image_pos_weight (8).
    Ranking and mining operate before gradient accumulation.
    """
    y = batch['label']
    base = outputs['base_logit']
    final = outputs['final_logit']
    loss_final = F.binary_cross_entropy_with_logits(final, y)
    loss_base = F.binary_cross_entropy_with_logits(base, y)
    text_g = image_g = final.new_zeros(())
    if args.ablation != 'without_mgr_complete':
        text_g = masked_bce_with_logits(outputs['text_ground_logit'], batch['token_target'], batch['token_valid'], pos_weight=args.text_pos_weight)
        image_g = masked_bce_with_logits(outputs['image_ground_logit'], batch['patch_target'], None, pos_weight=args.image_pos_weight)
    presence = final.new_zeros(())
    if args.ablation != 'without_haf_complete':
        presence = F.binary_cross_entropy_with_logits(outputs['text_presence_logit'], batch['text_manip']) + F.binary_cross_entropy_with_logits(outputs['image_presence_logit'], batch['image_manip'])
    state_target = batch['state']
    state_loss = F.cross_entropy(outputs['state_logits'], state_target)
    if args.ablation == 'without_dafo_complete':
        q = F.softmax(outputs['state_logits'], dim=-1)
        fake = (1 - q[:, 0]).clamp(0.0001, 1 - 0.0001)
        sb = F.binary_cross_entropy_with_logits(torch.log(fake) - torch.log1p(-fake), y)
        loss = 0.62 * loss_final + 0.12 * loss_base + 0.16 * state_loss + 0.07 * sb + 0.1 * presence + 0.06 * (text_g + image_g)
        return (loss, dict(final_cls=float(loss_final.detach()), base_cls=float(loss_base.detach()), state=float(state_loss.detach()), state_binary=float(sb.detach()), presence=float(presence.detach()), text_ground=float(text_g.detach()), image_ground=float(image_g.detach()), balanced_focal=0.0, dual_tail_hard=0.0, dual_margin=0.0, global_rank=0.0, text_rank=0.0, subtle_weight=1.0, dafo_active=0.0))
    per_sample = F.binary_cross_entropy_with_logits(final, y, reduction='none')
    true_prob = torch.where(y > 0.5, torch.sigmoid(final), torch.sigmoid(-final))
    focal_weight = torch.pow(1.0 - true_prob.detach(), 1.5)
    token_valid = batch['token_valid']
    text_extent = (batch['token_target'] * token_valid).sum(dim=-1) / token_valid.sum(dim=-1).clamp_min(1.0)
    text_subtle = 1.0 + 0.55 * (1.0 - text_extent.detach())
    text_related = (state_target == 1) | (state_target == 3)
    subtle_weight = torch.where(text_related, text_subtle, torch.ones_like(text_subtle))
    state_weight = torch.where(state_target == 0, torch.ones_like(y) * 1.15, torch.where(state_target == 1, torch.ones_like(y) * 1.35, torch.where(state_target == 2, torch.ones_like(y) * 0.95, torch.ones_like(y) * 0.9)))
    robust_sample_loss = per_sample * (0.45 + focal_weight) * subtle_weight * state_weight
    state_terms = []
    hard_terms = []
    for state_id in range(4):
        mask = state_target == state_id
        mask_f = mask.float()
        state_terms.append((robust_sample_loss * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        selected = robust_sample_loss[mask]
        if selected.numel() > 0:
            if state_id in (0, 1):
                hard_k = max(1, (selected.numel() + 1) // 2)
                hard_terms.append(torch.topk(selected, hard_k).values.mean())
    balanced_focal = torch.stack(state_terms).mean()
    if hard_terms:
        dual_tail_hard = torch.stack(hard_terms).mean()
    else:
        dual_tail_hard = robust_sample_loss.mean()
    dual_margin = final.new_zeros(())
    signed_margin = (2.0 * y - 1.0) * final
    target_margin = torch.where(state_target == 1, torch.ones_like(y) * 0.55, torch.where(state_target == 0, torch.ones_like(y) * 0.35, torch.ones_like(y) * 0.2))
    margin_loss = F.softplus(target_margin - signed_margin)
    margin_weight = torch.where(state_target == 1, torch.ones_like(y) * 1.35, torch.where(state_target == 0, torch.ones_like(y) * 1.15, torch.ones_like(y)))
    dual_margin = (margin_loss * margin_weight).mean()
    state_binary = final.new_zeros(())
    if args.ablation != 'without_haf_complete':
        state_prob = F.softmax(outputs['state_logits'], dim=-1)
        p_fake_state = torch.clamp(1.0 - state_prob[:, 0], min=0.0001, max=1.0 - 0.0001)
        state_fake_logit = torch.log(p_fake_state) - torch.log1p(-p_fake_state)
        state_binary = F.binary_cross_entropy_with_logits(state_fake_logit, y)
    global_rank = torch.tensor(0.0, device=y.device)
    pos = final[y > 0.5]
    neg = final[y <= 0.5]
    if pos.numel() > 0 and neg.numel() > 0:
        global_rank = F.softplus(-(pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
    text_rank = torch.tensor(0.0, device=y.device)
    text_logits = final[state_target == 1]
    real_logits = final[state_target == 0]
    if text_logits.numel() > 0 and real_logits.numel() > 0:
        text_rank = F.softplus(-(text_logits.unsqueeze(1) - real_logits.unsqueeze(0))).mean()
    loss = 0.62 * loss_final + 0.12 * loss_base + 0.3 * balanced_focal + 0.14 * dual_tail_hard + 0.16 * dual_margin + 0.16 * state_loss + 0.07 * state_binary + 0.1 * presence + 0.06 * (text_g + image_g) + 0.08 * global_rank + 0.1 * text_rank
    return (loss, {'final_cls': float(loss_final.detach()), 'base_cls': float(loss_base.detach()), 'balanced_focal': float(balanced_focal.detach()), 'dual_tail_hard': float(dual_tail_hard.detach()), 'dual_margin': float(dual_margin.detach()), 'state': float(state_loss.detach()), 'state_binary': float(state_binary.detach()), 'presence': float(presence.detach()), 'text_ground': float(text_g.detach()), 'image_ground': float(image_g.detach()), 'global_rank': float(global_rank.detach()), 'text_rank': float(text_rank.detach()), 'focal_weight': float(focal_weight.mean().detach()), 'subtle_weight': float(subtle_weight.mean().detach()), 'dafo_active': 1.0})

def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def build_optimizer(model, args):
    backbone, heads = ([], [])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('clip.'):
            backbone.append(p)
        else:
            heads.append(p)
    groups = []
    if heads:
        groups.append({'params': heads, 'lr': args.lr})
    if backbone:
        groups.append({'params': backbone, 'lr': args.clip_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
