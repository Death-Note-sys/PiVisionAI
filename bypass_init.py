with open('app/modules/ai_identify/controller.py', 'r') as f:
    code = f.read()

code = code.replace('if des is not None:', 'if True:')

with open('app/modules/ai_identify/controller.py', 'w') as f:
    f.write(code)
