# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['pocker.py'],
    pathex=[],
    binaries=[],
    datas=[('nothing.ttf', '.'), ('card_detector/model.pt', 'card_detector'), ('denomination_detector/model.pth', 'denomination_detector'), ('suit_detector/model.pth', 'suit_detector')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pocker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
