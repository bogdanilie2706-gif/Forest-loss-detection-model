#!/usr/bin/env bash
# Starea antrenării, pe scurt: proces principal, progres, ultima epocă, GPU.
# Folosit de `make status`.

pids=$(pgrep -f "[s]rc/train.py" || true)

# Procesul principal = proces python al cărui părinte nu e alt proces train.py.
# Workerii DataLoader sunt copii ai lui și se ignoră.
main=""
for p in $pids; do
    [[ "$(ps -o comm= -p "$p" 2>/dev/null)" == python* ]] || continue
    parent=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    if ! grep -qx "$parent" <<< "$pids"; then
        main=$p
        break
    fi
done

if [[ -z "$main" ]]; then
    echo "Nu rulează nicio antrenare."
    exit 0
fi

args=$(tr '\0' ' ' < "/proc/$main/cmdline")
name=$(sed -n 's/.*--name \([^ ]*\).*/\1/p' <<< "$args")
arch=$(sed -n 's/.*--arch \([^ ]*\).*/\1/p' <<< "$args")
total=$(sed -n 's/.*--epochs \([0-9]*\).*/\1/p' <<< "$args")
log="logs/$name.txt"

secs=$(ps -o etimes= -p "$main" | tr -d ' ')
elapsed=$(printf '%d:%02d:%02d' $((secs / 3600)) $((secs % 3600 / 60)) $((secs % 60)))

echo "Antrenare în curs: ${name:-?} (${arch:-cnn})"
echo "  PID $main · rulează de $elapsed"

if [[ -f "$log" ]]; then
    done_ep=$(grep -c '^Epoca' "$log")
    echo -n "  Progres: $done_ep/${total:-?} epoci"
    if [[ "$done_ep" -gt 0 && -n "$total" ]]; then
        eta=$(( secs * (total - done_ep) / done_ep / 60 ))
        echo " · ETA ~${eta} min (estimare)"
    else
        echo " · prima epocă încă nu s-a terminat"
    fi
    last=$(grep '^Epoca' "$log" | tail -n 1)
    [[ -n "$last" ]] && echo "  Ultima epocă: $last"
else
    echo "  (fără log: rulare în prim-plan sau cu alt nume)"
fi

if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits |
        awk -F', ' '{printf "  GPU: %s%% utilizare · %s/%s MiB\n", $1, $2, $3}'
fi