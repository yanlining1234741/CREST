import json, glob, os
base = '/sda/yxyang/qsba_workspace/mbeir_aligned'

GENIUS = {
    'visualnews': (27.4, 'R@5'),
    'visualnews_task3': (28.4, 'R@5'),
    'mscoco': (68.0, 'R@5'),
    'mscoco_task3': (91.1, 'R@5'),
    'flickr': (92.0, 'R@5'),
    'fashion200k_task0': (16.2, 'R@10'),
    'nights_task4': (30.2, 'R@5'),
}

# 找所有 dataset + config 组合
results = []
for ds_dir in sorted(glob.glob(f'{base}/outputs/*/')):
    ds = os.path.basename(os.path.dirname(ds_dir))
    if ds not in GENIUS: continue
    # 找 cosine eval
    for ev_file in sorted(glob.glob(f'{ds_dir}/eval_*cosine*.json')):
        cfg = os.path.basename(ev_file).replace('eval_', '').replace('_cosine.json', '').replace('_cosine_fix.json', '').replace('.json', '')
        try:
            cos = json.load(open(ev_file))
        except: continue
        # 找对应 CE
        ce_dir = f'{base}/cross_encoder/{ds}_{cfg}'
        ce_eval = f'{ce_dir}/eval_cross_encoder.json'
        ce = json.load(open(ce_eval)) if os.path.exists(ce_eval) else None
        
        g_val, g_metric = GENIUS[ds]
        metric_key = g_metric.replace('R@', 'recall@').replace('@', '@').lower()  # "recall@5"
        
        for B in [1, 3, 5]:
            ck = f'{metric_key}|B={B}'
            if ck not in cos: continue
            cand = cos.get(f'candidates@B={B}', 0)
            cos_r = cos[ck] * 100
            ce_r = ce[ck] * 100 if ce and ck in ce else None
            results.append({
                'ds': ds, 'cfg': cfg, 'B': B, 'cand': cand,
                'metric': g_metric, 'cos': cos_r, 'ce': ce_r,
                'delta': (ce_r - cos_r) if ce_r is not None else None,
                'genius': g_val,
            })

print(f"\n{'Dataset':<22} {'Cfg':<12} {'B':>2} {'cand':>8} {'metric':>5} {'cos':>7} {'CE':>7} {'ΔCE':>7} {'GENIUS':>7} {'ΔvsG':>7}")
print("=" * 105)
for r in results:
    ce_s = f"{r['ce']:.2f}" if r['ce'] is not None else "  -  "
    delta_s = f"{r['delta']:+.2f}" if r['delta'] is not None else "  -  "
    print(f"{r['ds']:<22} {r['cfg']:<12} {r['B']:>2} {r['cand']:>8.0f} {r['metric']:>5} {r['cos']:>7.2f} {ce_s:>7} {delta_s:>7} {r['genius']:>7.1f} {r['cos']-r['genius']:>+7.2f}")
