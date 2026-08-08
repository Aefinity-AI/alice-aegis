@echo -off
rem A.L.I.C.E. legacy-firmware fallback.
rem Older/quirky UEFI implementations that ignore the removable-media
rem default boot path (\EFI\BOOT\BOOTX64.EFI) drop into the EFI shell
rem instead; the shell auto-runs this script and chain-loads the engine.
rem Lost from the Jul-12 image rebuild because build_usb_img.sh never
rem copied it -- it is a tracked artifact now. Do not remove.
echo A.L.I.C.E. fallback loader: searching for BOOTX64.EFI...
for %i in fs0 fs1 fs2 fs3
    if exist %i:\EFI\BOOT\BOOTX64.EFI then
        %i:
        \EFI\BOOT\BOOTX64.EFI
    endif
endfor
echo startup.nsh: BOOTX64.EFI not found on fs0-fs3.
