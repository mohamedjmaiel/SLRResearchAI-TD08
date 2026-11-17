from pathlib import Path

import streamlit as st

from slrresearch.orchestrator import full_pipeline_run, full_supervisor_run
from slrresearch.quality_agent import build_llm_judge, evaluate_rules
from slrresearch.td08_runtime import artifact_writer, build_artifact_workers, build_state, load_td08_artifacts


st.set_page_config(page_title="TD08 — QualityAgent et orchestration", page_icon=":material/fact_check:", layout="wide")
ROOT = Path(__file__).parent
OBJECTIVE = "Contrôler la fidélité d'une note SLR avant sa livraison."


@st.cache_data
def load_inputs():
    return load_td08_artifacts(ROOT)


def quality_rows(report):
    return [{"Affirmation": f.claim_id, "Statut": f.status, "Justification": f.reason} for f in report.findings]


def report_evolution(before: dict, after: dict):
    """Compare le rapport initiale et la note réparée, affirmation par affirmation."""
    before_by_id = {item.get("id"): item for item in before.get("claims", [])}
    after_by_id = {item.get("id"): item for item in after.get("claims", [])}
    rows = []
    for claim_id in sorted(set(before_by_id) | set(after_by_id)):
        old = before_by_id.get(claim_id, {})
        new = after_by_id.get(claim_id, {})
        rows.append({
            "ID": claim_id,
            "Texte initial": old.get("text", "—"),
            "Texte après révision": new.get("text", "—"),
            "Preuves initiales": ", ".join(old.get("evidence_ids", [])) or "—",
            "Preuves après révision": ", ".join(new.get("evidence_ids", [])) or "—",
            "Modifiée": "oui" if old != new else "non",
        })
    return rows


AGENT_LABELS = {
    "search": "SearchAgent",
    "screening": "ScreeningAgent",
    "rag": "RAGAgent",
    "writing": "WritingAgent",
    "quality": "QualityAgent",
}


def execution_rows(result):
    """Trace lisible : un événement par agent et par handoff."""
    rows = []
    for index, agent in enumerate(result.state.history, start=1):
        rows.append({"Étape": index, "Agent exécuté": AGENT_LABELS.get(agent, agent), "État": "terminé"})
    for index, handoff in enumerate(result.handoffs, start=len(rows) + 1):
        rows.append({
            "Étape": index,
            "Agent exécuté": f"{AGENT_LABELS.get(handoff.source, handoff.source)} → {AGENT_LABELS.get(handoff.target, handoff.target)}",
            "État": f"handoff {handoff.status}, budget={handoff.budget_left}",
        })
    return rows


def artifact_rows(result):
    rows = []
    for name, artifact in result.state.artifacts.items():
        rows.append({
            "Agent": AGENT_LABELS.get(name, name),
            "Artefact produit": f"{len(artifact)} champs",
            "Champs principaux": ", ".join(list(artifact)[:5]),
        })
    return rows


def architecture_dot():
    return r'''digraph G {
      graph [rankdir=LR, bgcolor="transparent", pad="0.2"];
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#334155", penwidth=1.3];
      edge [fontname="Helvetica", fontsize=10, color="#64748b", arrowsize=0.7];
      search [label="1  SearchAgent\narticles", fillcolor="#dbeafe"];
      screening [label="2  ScreeningAgent\ninclude / exclude", fillcolor="#dcfce7"];
      rag [label="3  RAGAgent\npreuves vérifiées", fillcolor="#fef3c7"];
      writing [label="4  WritingAgent\nrapport", fillcolor="#fce7f3"];
      quality [label="5  QualityAgent\nACCEPT / REVISE", fillcolor="#ede9fe"];
      search -> screening -> rag -> writing -> quality;
      quality -> rag [label="REVISE : preuve", color="#dc2626", fontcolor="#dc2626"];
      quality -> finish [label="ACCEPT", color="#16a34a", fontcolor="#16a34a", style=dashed];
      finish [label="FINISH", shape=doublecircle, fillcolor="#bbf7d0"];
    }'''


def pattern_dot(pattern):
    """Schéma conceptuel du pattern, avant l'observation de la trace réelle."""
    if pattern == "pipeline":
        return r'''digraph G {
          graph [rankdir=LR, bgcolor="transparent"];
          node [shape=box, style="rounded,filled", fontname="Helvetica", color="#334155"];
          search [label="Search", fillcolor="#dbeafe"]; screening [label="Screening", fillcolor="#dcfce7"];
          rag [label="RAG", fillcolor="#fef3c7"]; writing [label="Writing", fillcolor="#fce7f3"];
          quality [label="Quality", fillcolor="#ede9fe"]; stop [label="STOP", shape=doublecircle, fillcolor="#fecaca"];
          search -> screening -> rag -> writing -> quality -> stop;
        }'''
    return r'''digraph G {
      graph [rankdir=LR, bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fontname="Helvetica", color="#334155"];
      search [label="Search", fillcolor="#dbeafe"]; screening [label="Screening", fillcolor="#dcfce7"];
      rag [label="RAG", fillcolor="#fef3c7"]; writing [label="Writing", fillcolor="#fce7f3"];
      quality [label="Quality", fillcolor="#ede9fe"]; finish [label="FINISH", shape=doublecircle, fillcolor="#bbf7d0"];
      search -> screening -> rag -> writing -> quality;
      quality -> rag [label="REVISE : preuve", color="#dc2626", fontcolor="#dc2626"];
      rag -> writing [label="artefact corrigé", color="#dc2626", fontcolor="#dc2626"];
      quality -> finish [label="ACCEPT", color="#16a34a", fontcolor="#16a34a", style=dashed];
    }'''


def result_dot(result):
    """Graphe de l'exécution réellement observée, pas un schéma statique."""
    colors = {"search": "#dbeafe", "screening": "#dcfce7", "rag": "#fef3c7", "writing": "#fce7f3", "quality": "#ede9fe"}
    lines = ['digraph G {', 'graph [rankdir=LR, bgcolor="transparent", pad="0.2"];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#334155"];']
    seen = set()
    for index, agent in enumerate(result.state.history, start=1):
        seen.add(agent)
        label = f"{index}. {AGENT_LABELS.get(agent, agent)}"
        lines.append(f'{agent} [label="{label}", fillcolor="{colors.get(agent, "#e2e8f0")}"];')
    lines.append('finish [label="FINISH", shape=doublecircle, fillcolor="#bbf7d0"];')
    lines.append('escalate [label="ESCALATE", shape=octagon, fillcolor="#fecaca"];')
    for transition in result.transitions:
        source, target = transition.split("→", 1)
        target_id = target if target in {"finish", "escalate"} else target
        color = "#dc2626" if source == "quality" and target in {"search", "screening", "rag", "writing"} else "#64748b"
        lines.append(f'{source} -> {target_id} [color="{color}"];')
    lines.append('}')
    return "\n".join(lines)


def show_result(title, result):
    with st.container(border=True):
        st.subheader(title)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Arrêt", result.status)
        c2.metric("Verdict final", result.state.quality.verdict)
        c3.metric("Score", f"{result.state.quality.score:.0%}")
        c4.metric("Agents appelés", len(result.state.history))
        c5.metric("Appels LLM-juge", result.state.quality.llm_calls)
        st.markdown("**Agents effectivement appelés — ordre d'exécution**")
        st.code("  →  ".join(AGENT_LABELS.get(agent, agent) for agent in result.state.history), language="text")
        st.graphviz_chart(result_dot(result), use_container_width=True)
        st.markdown("**Trace d'exécution et handoffs**")
        st.dataframe(execution_rows(result), hide_index=True, use_container_width=True)
        st.markdown("**Artefacts dans l'état partagé**")
        st.dataframe(artifact_rows(result), hide_index=True, use_container_width=True)
        with st.expander("Voir l'état partagé final"):
            st.json(result.state.model_dump(mode="json"))
        with st.expander("Voir le détail de chaque artefact"):
            for name, artifact in result.state.artifacts.items():
                st.markdown(f"**{AGENT_LABELS.get(name, name)}**")
                st.json(artifact)
        st.markdown("**Décision d'orchestration**")
        st.write(" → ".join(result.transitions))
        st.dataframe(quality_rows(result.state.quality), hide_index=True)
        if result.state.quality.feedback:
            st.warning("Feedback : " + " | ".join(result.state.quality.feedback))


evidence, required_rqs, draft, repaired = load_inputs()
st.title("TD08 — QualityAgent et orchestration")
st.warning("Le corpus est pédagogique : la note n'est pas une SLR publiable.")
st.write("Partie I : contrôler une note affirmation par affirmation. Partie II : comparer pipeline fixe et superviseur.")
with st.container(border=True):
    st.metric("Preuves héritées du TD06", len(evidence))
    st.metric("Questions de recherche", len(required_rqs))
    st.caption("L'artefact de rédaction provient du TD07 et contient volontairement une affirmation insuffisamment étayée.")

st.header("Phase 1 — QualityAgent seul")
st.info("Commencez ici : le QualityAgent mesure la qualité de la note avant toute orchestration multi-agent.")
with st.form("quality_form", border=True):
    mode = st.segmented_control("Évaluateur", ["Règles déterministes", "LLM-juge réel"], default="Règles déterministes")
    run_quality = st.form_submit_button("1 — Évaluer la note TD07", type="primary", icon=":material/fact_check:")

if run_quality:
    try:
        evaluator = lambda report: evaluate_rules(report, evidence, required_rqs)
        if mode == "LLM-juge réel":
            from slrresearch.config import Settings, build_model
            judge = build_llm_judge(build_model(Settings()))
            evaluator = lambda report: judge(report, evaluate_rules(report, evidence, required_rqs))
        st.session_state.quality_result = {
            "mode": mode,
            "draft": evaluator(draft),
            "repaired": evaluator(repaired),
        }
        st.session_state.orchestration_result = None
    except Exception as exc:
        st.error(f"Évaluation impossible : {exc}")

quality = st.session_state.get("quality_result")
if quality:
    left, right = st.columns(2)
    with left:
        st.subheader("Note initiale du TD07")
        st.metric("Verdict", quality["draft"].verdict)
        st.metric("Fidélité", f"{quality['draft'].faithfulness:.0%}")
        st.dataframe(quality_rows(quality["draft"]), hide_index=True)
        if quality["draft"].feedback:
            st.warning("Feedback : " + " | ".join(quality["draft"].feedback))
    with right:
        st.subheader("Note après correction")
        st.metric("Verdict", quality["repaired"].verdict)
        st.metric("Fidélité", f"{quality['repaired'].faithfulness:.0%}")
        st.dataframe(quality_rows(quality["repaired"]), hide_index=True)
    with st.expander("Voir comment la note a évolué", expanded=True):
        st.caption("La correction ne consiste pas seulement à changer le verdict : elle doit modifier l'affirmation ou ses preuves.")
        st.dataframe(report_evolution(draft, repaired), hide_index=True, use_container_width=True)
        st.markdown(
            f"**Évolution globale :** score `{quality['draft'].score:.0%}` → `{quality['repaired'].score:.0%}`, "
            f"fidélité `{quality['draft'].faithfulness:.0%}` → `{quality['repaired'].faithfulness:.0%}`, "
            f"verdict `{quality['draft'].verdict}` → `{quality['repaired'].verdict}`."
        )
        st.warning(
            "**Cas A5 :** l'affirmation initiale « La précision dépasse celle des experts. » "
            "cite E8, mais E8 parle uniquement de calibration et de validation locale du prompt. "
            "La preuve ne dit donc pas que la précision dépasse celle des experts. "
            "La version corrigée A5R « Le prompt doit être calibré et contrôlé dans le domaine avant déploiement. » "
            "reprend directement le contenu de E8 : elle devient vérifiable."
        )
    st.success("Conclusion de la phase 1 : le contrôle qualité détecte une affirmation insuffisamment soutenue et rend le feedback traçable.")

    st.header("Phase 2 — Orchestration multi-agent")
    st.info("Cette phase n'est disponible qu'après l'évaluation du QualityAgent. Elle montre les cinq agents, les artefacts partagés, les handoffs et la différence entre pipeline et superviseur.")
    st.subheader("Architecture cible")
    st.caption("Les flèches représentent les handoffs autorisés. La flèche rouge est la boucle de révision du superviseur.")
    st.graphviz_chart(architecture_dot(), use_container_width=True)
    with st.form("orchestration_form", border=True):
        budget = st.slider("Budget maximal de handoffs", 5, 10, 8)
        run_orchestration = st.form_submit_button("2 — Comparer pipeline et superviseur", type="primary", icon=":material/account_tree:")
    if run_orchestration:
        try:
            # Le même juge que celui choisi en phase 1 est utilisé pendant
            # chaque passage au QualityAgent dans les deux orchestrations.
            evaluator = lambda report: evaluate_rules(report, evidence, required_rqs)
            if quality["mode"] == "LLM-juge réel":
                from slrresearch.config import Settings, build_model
                judge = build_llm_judge(build_model(Settings()))
                evaluator = lambda report: judge(report, evaluate_rules(report, evidence, required_rqs))
            result = {
                # Un worker neuf par expérience : le compteur du WritingAgent
                # ne doit pas influencer la comparaison.
                "pipeline": full_pipeline_run(
                    build_state(draft, evidence, OBJECTIVE, budget),
                    build_artifact_workers(ROOT, evidence, draft, repaired), evaluator,
                ),
                "supervisor": full_supervisor_run(
                    build_state(draft, evidence, OBJECTIVE, budget),
                    build_artifact_workers(ROOT, evidence, draft, repaired), evaluator,
                    max_handoffs=budget,
                ),
                "judge_mode": quality["mode"],
            }
            st.session_state.orchestration_result = result
        except Exception as exc:
            st.error(f"Orchestration impossible : {exc}")

orchestration = st.session_state.get("orchestration_result")
if quality and orchestration:
    st.info(f"QualityAgent utilisé dans les deux expériences — mode : **{orchestration.get('judge_mode', 'Règles déterministes')}**. Le compteur `Appels LLM-juge` permet de vérifier les appels réels.")
    st.subheader("Comparer les deux modes d'orchestration")
    st.dataframe([
        {"Pattern": "Pipeline fixe", "Décision": "ordre prédéfini", "Boucle": "non", "Feedback Quality→Writing": "non", "Arrêt attendu": "REVISE → ESCALATE"},
        {"Pattern": "Supervisor–Workers", "Décision": "superviseur", "Boucle": "oui, bornée", "Feedback Quality→Writing": "oui", "Arrêt attendu": "ACCEPT → FINISH"},
    ], hide_index=True, use_container_width=True)
    pipeline_tab, supervisor_tab = st.tabs(["Pipeline fixe", "Supervisor–Workers"])
    with pipeline_tab:
        st.caption("Le pipeline applique un ordre fixe. QualityAgent observe le rapport mais ne peut pas rappeler WritingAgent.")
        st.graphviz_chart(pattern_dot("pipeline"), use_container_width=True)
        show_result("Trace du pipeline fixe", orchestration["pipeline"])
    with supervisor_tab:
        st.caption("Le superviseur lit le verdict et choisit : FINISH, ESCALATE ou retour vers WritingAgent.")
        st.graphviz_chart(pattern_dot("supervisor"), use_container_width=True)
        show_result("Trace du Supervisor–Workers", orchestration["supervisor"])
        supervisor = orchestration["supervisor"]
        if "writing_initial" in supervisor.state.artifacts and "writing" in supervisor.state.artifacts:
            with st.expander("Évolution de la note dans la boucle de supervision"):
                st.write("Le premier artefact est produit avant le contrôle qualité ; le second après le retour vers l'agent responsable du problème.")
                st.dataframe(
                    report_evolution(supervisor.state.artifacts["writing_initial"], supervisor.state.artifacts["writing"]),
                    hide_index=True,
                    use_container_width=True,
                )
    st.header("Interprétation finale")
    st.markdown("""- Le pipeline fixe est prévisible mais ne répare pas une note.
- Le superviseur diagnostique le problème et route ici vers le RAGAgent avant le WritingAgent.
- Le coût supplémentaire est visible dans les actions, les transitions et le budget.
- Une architecture multi-agent est justifiée seulement si le gain de qualité est mesurable.
""")
