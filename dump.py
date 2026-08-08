import os

dirs_to_scan = [
    '/home/killboxincorporated/aegis-core',
    '/home/killboxincorporated/aegis-uefi',
]

with open('/home/killboxincorporated/alice_unikernel.txt', 'w') as out:
    out.write("========================================\n")
    out.write("A.L.I.C.E. UNIKERNEL V11 SOURCE CODE DUMP\n")
    out.write("========================================\n\n")

    for dir_path in dirs_to_scan:
        for root, _, files in os.walk(dir_path):
            if 'target' in root:
                continue
            for file in files:
                if file.endswith('.rs') or file.endswith('.toml') or file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                        out.write(f"File: {filepath}\n")
                        out.write("-" * 40 + "\n")
                        out.write(content)
                        out.write("\n\n")
                    except Exception:
                        pass
