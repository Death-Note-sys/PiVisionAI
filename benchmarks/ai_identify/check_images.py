import cv2, os

BENCH = os.path.join('benchmarks', 'ai_identify')
for root, dirs, files in os.walk(BENCH):
    for fname in sorted(files):
        if fname == '.gitkeep':
            continue
        p = os.path.join(root, fname)
        rel = p.replace(BENCH + os.sep, '')
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            print(f'  {rel:<45} {w}x{h}')
        else:
            print(f'  UNREADABLE: {rel}')
