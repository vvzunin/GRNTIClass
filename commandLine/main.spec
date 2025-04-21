# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py', 'prediction.py'],
    pathex=[],
    binaries=[],
    datas=[ ('my_grnti1_int.json', '.'), 
            ('my_grnti2_int.json', '.'), 
            ('GRNTI_1_ru.json', '.'), 
            ('GRNTI_2_ru.json', '.'),
            ('config.json', '.'),
            ('models/bert2', 'models/bert2'),
            ('doc', 'doc'),
            ('examples', 'examples')],
    hiddenimports=[],
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
