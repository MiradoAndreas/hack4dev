#!/usr/bin/env bash
# Run from any directory; install only the explicitly selected project skills.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SCIENTIFIC_SKILLS=(exploratory-data-analysis scikit-learn transformers pytorch-lightning shap scientific-slides)
SUPERPOWER_SKILLS=(systematic-debugging verification-before-completion writing-plans)
HF_SKILLS=(huggingface-llm-trainer huggingface-gradio)
# The user selected huggingface-llm-trainer as the intended trainer skill.

verify_skills() {
    python3 scripts/verify-agent-skills.py \
        "${SCIENTIFIC_SKILLS[@]}" "${SUPERPOWER_SKILLS[@]}" "${HF_SKILLS[@]}"
}

case "${1:-install}" in
    --check) verify_skills; exit $? ;;
    install) ;;
    *) printf 'Usage: bash scripts/setup-agent-skills.sh [install|--check]\n' >&2; exit 2 ;;
esac

command -v python3 >/dev/null || { printf 'Python 3 is required for verification.\n' >&2; exit 1; }
command -v npx >/dev/null || { printf 'Node.js and npx are required.\n' >&2; exit 1; }
if [[ -d .agents && ! -w .agents ]]; then
    printf 'Project .agents directory is read-only. Run from a terminal with write access.\n' >&2
    exit 1
fi

install_missing() {
    local repo="$1"
    shift
    local skill
    local pending=()
    for skill in "$@"; do
        if [[ -e ".agents/skills/$skill" || -L ".agents/skills/$skill" ]]; then
            # Validate existing installations; never overwrite them automatically.
            python3 scripts/verify-agent-skills.py "$skill"
        else
            pending+=("$skill")
        fi
    done
    if ((${#pending[@]})); then
        # No -g: scope installation to the repository and target Codex explicitly.
        npx --yes skills add "$repo" --skill "${pending[@]}" --agent codex --yes
    fi
}

install_missing K-Dense-AI/scientific-agent-skills "${SCIENTIFIC_SKILLS[@]}"
install_missing obra/superpowers "${SUPERPOWER_SKILLS[@]}"
install_missing huggingface/skills "${HF_SKILLS[@]}"

verify_skills
printf 'All eleven selected skill metadata checks passed.\n'
printf 'Start a new agent turn/session and confirm the available-skills list.\n'
