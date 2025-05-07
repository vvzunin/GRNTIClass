# -*- mode: python ; coding: utf-8 -*-


a = Analysis([
    '../config.py',
    '../help_message.py',
    '../main.py',
    '../messages.py',
    '../prediction.py'],
  pathex=[],
  binaries=[],
  datas=[
    ('../dicts', 'dicts'),
    ('../config.json', '.'),
    ('../prog.json', '.'),
    ('../models/bert2', 'models/bert2'),
    ('../doc', 'doc'),
    ('../examples', 'examples'),
    ('../run-container.sh', '.'),
    ('../src/backend', 'src/backend'),
    ('../src/frontend/config.json', 'src/frontend'),],
  hiddenimports=[
    'transformers.models.bert',
    'transformers.models.bert.configuration_bert',
    'transformers.models.bert.modeling_bert',
    'peft',
    'peft.tuners.lora',],
  hookspath=[],
  hooksconfig={},
  runtime_hooks=[],
  excludes=[],
  onedir=False
)
pyz = PYZ(a.pure)

exe = EXE(
  pyz,
  a.scripts,
  [],
  exclude_binaries=False,
  name='GRNTIClass',
  debug=False,
  bootloader_ignore_signals=False,
  strip=False,
  upx=True,
  console=True,
  disable_windowed_traceback=False,
  argv_emulation=False,
  target_arch=None,
  codesign_identity=None,
  entitlements_file=None,
  contents_directory='.'
)
coll = COLLECT(
  exe,
  a.binaries,
  a.datas,
  strip=False,
  upx=True,
  upx_exclude=[],
  name='GRNTIClass'
)
