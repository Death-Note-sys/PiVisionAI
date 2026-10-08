with open('app/modules/ai_identify/controller.py', 'r') as f:
    code = f.read()

target = 'self.teach_status = "Untaught"'
replacement = target + '\n            self.save_data({"teach_status": "Untaught", "good_count": 0, "bad_count": 0})'
if 'self.save_data({"teach_status": "Untaught"' not in code:
    code = code.replace(target, replacement)

with open('app/modules/ai_identify/controller.py', 'w') as f:
    f.write(code)
