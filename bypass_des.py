with open('app/modules/ai_identify/controller.py', 'r') as f:
    code = f.read()

code = code.replace('if des is None or len(kp) < 0:', 'if False:')

with open('app/modules/ai_identify/controller.py', 'w') as f:
    f.write(code)
