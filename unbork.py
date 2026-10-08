with open('app/modules/ai_identify/controller.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'self.save_data({"teach_status": "Untaught", "good_count": 0, "bad_count": 0})' in line:
        if 'def reset_teaching' not in "".join(lines[max(0, i-10):i]):
            lines[i] = ""

with open('app/modules/ai_identify/controller.py', 'w') as f:
    f.writelines(lines)
