#!/usr/bin/env zsh
set -e

REF="step5_input.gro"
PREV_GRO="step6.0_em.gro"

for i in 1 2 3 4 5 6; do
    STEP="step6.${i}_equilibration"
    echo ""
    echo "========================================="
    echo ">>> Starting ${STEP}"
    echo "========================================="

    if [ $i -eq 1 ]; then
        gmx grompp \
          -f ${STEP}.mdp \
          -c ${PREV_GRO} \
          -r ${REF} \
          -p topol.top \
          -n index.ndx \
          -o ${STEP}.tpr \
          -maxwarn 2
    else
        PREV_CPT="step6.$((i-1))_equilibration.cpt"
        gmx grompp \
          -f ${STEP}.mdp \
          -c ${PREV_GRO} \
          -t ${PREV_CPT} \
          -r ${REF} \
          -p topol.top \
          -n index.ndx \
          -o ${STEP}.tpr \
          -maxwarn 2
    fi

    gmx mdrun -v -deffnm ${STEP} -ntmpi 1 -ntomp 8

    PREV_GRO="${STEP}.gro"
    echo ">>> DONE: ${STEP}"
done

echo ""
echo "=== ALL EQUILIBRATION STEPS DONE ==="
