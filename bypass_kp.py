with open('app/modules/ai_identify/controller.py', 'r') as f:
    code = f.read()

code = code.replace('len(kp) < 10', 'len(kp) < 0')

with open('app/modules/ai_identify/controller.py', 'w') as f:
    f.write(code)
