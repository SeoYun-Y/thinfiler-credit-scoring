import os

root = r"c:\Users\tehun\Desktop\multicamp\프로젝트\creditscore"
for dirpath, dirnames, filenames in os.walk(root):
    depth = dirpath.replace(root, "").count(os.sep)
    if depth > 2:  # 너무 깊이 들어가지 않도록 제한
        continue
    indent = "  " * depth
    print(f"{indent}{os.path.basename(dirpath)}/")
    for f in filenames[:10]:
        print(f"{indent}  - {f}")