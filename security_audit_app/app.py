import sys
import os
import asyncio
from typing import TypedDict, Annotated, List
from operator import add
from dotenv import load_dotenv
import gradio as gr
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# ─────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────
class AgentState(TypedDict):
    infrastructure: str
    audit_brief: str
    audit_reports: Annotated[List[str], add]
    final_report: str

# ─────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────
def orchestrator(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""You are a senior cloud security audit coordinator.
        Analyze the infrastructure description and prepare a clear concise audit brief
        that will guide three specialist security auditors.
        Identify key systems, components, and risk areas to examine. Maximum 150 words."""),
        HumanMessage(content=f"Infrastructure to audit:\n{state['infrastructure']}")
    ])
    return {"audit_brief": response.content}

def iam_audit(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""You are an IAM security specialist.
        Audit the infrastructure for Identity and Access Management risks.
        Cover: authentication, authorization, least privilege, MFA,
        service accounts, and role assignments.
        Be specific and actionable. Maximum 200 words."""),
        HumanMessage(content=f"Audit brief:\n{state['audit_brief']}\n\nInfrastructure:\n{state['infrastructure']}")
    ])
    return {"audit_reports": [f"### IAM Security Audit\n{response.content}"]}

def network_audit(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""You are a network security specialist.
        Audit the infrastructure for network security risks.
        Cover: firewall rules, network segmentation, VPC configuration,
        exposed ports, traffic encryption, and DDoS protection.
        Be specific and actionable. Maximum 200 words."""),
        HumanMessage(content=f"Audit brief:\n{state['audit_brief']}\n\nInfrastructure:\n{state['infrastructure']}")
    ])
    return {"audit_reports": [f"### Network Security Audit\n{response.content}"]}

def compliance_audit(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""You are a cloud compliance specialist.
        Audit the infrastructure for compliance and governance risks.
        Cover: regulatory requirements, data protection, logging,
        monitoring, patch management, and CIS/NIST benchmarks.
        Be specific and actionable. Maximum 200 words."""),
        HumanMessage(content=f"Audit brief:\n{state['audit_brief']}\n\nInfrastructure:\n{state['infrastructure']}")
    ])
    return {"audit_reports": [f"### Compliance & Governance Audit\n{response.content}"]}

def synthesizer(state: AgentState) -> dict:
    all_reports = "\n\n".join(state["audit_reports"])
    response = llm.invoke([
        SystemMessage(content="""You are a senior security audit manager.
        Combine the three specialist audit reports into one comprehensive
        executive-level security assessment.
        Highlight critical findings, common themes, and provide a
        prioritized remediation list. Maximum 400 words."""),
        HumanMessage(content=f"""
Audit Brief:
{state['audit_brief']}

Individual Audit Reports:
{all_reports}

Synthesize into a final executive security assessment.""")
    ])
    return {"final_report": response.content}

# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("iam_audit", iam_audit)
workflow.add_node("network_audit", network_audit)
workflow.add_node("compliance_audit", compliance_audit)
workflow.add_node("synthesizer", synthesizer)

workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", "iam_audit")
workflow.add_edge("orchestrator", "network_audit")
workflow.add_edge("orchestrator", "compliance_audit")
workflow.add_edge("iam_audit", "synthesizer")
workflow.add_edge("network_audit", "synthesizer")
workflow.add_edge("compliance_audit", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# ─────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────
def run_security_audit(infrastructure_input: str) -> tuple:
    if not infrastructure_input.strip():
        return "Please describe your infrastructure.", "", "", ""
    try:
        result = app.invoke({
            "infrastructure": infrastructure_input,
            "audit_brief": "",
            "audit_reports": [],
            "final_report": ""
        })
        brief = result.get("audit_brief", "")
        reports = result.get("audit_reports", [])
        iam = next((r for r in reports if "IAM" in r), "No IAM report")
        network = next((r for r in reports if "Network" in r), "No network report")
        compliance = next((r for r in reports if "Compliance" in r), "No compliance report")
        final = result.get("final_report", "No final report generated")
        individual = f"{iam}\n\n---\n\n{network}\n\n---\n\n{compliance}"
        return brief, individual, final
    except Exception as e:
        return f"Error: {str(e)}", "", ""

with gr.Blocks(title="Cloud Security Audit Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Cloud Security Audit Agent
    ### Powered by 3 Parallel AI Security Specialists
    *Describe your cloud infrastructure and get an instant multi-agent security assessment*
    """)
    with gr.Row():
        with gr.Column():
            infra_input = gr.Textbox(
                label="Describe your cloud infrastructure",
                placeholder="e.g. Kubernetes cluster on Azure with public API endpoint, admin service accounts with wildcard permissions, no MFA enforced...",
                lines=4
            )
            audit_btn = gr.Button("Run Security Audit", variant="primary", size="lg")
    gr.Markdown("---")
    with gr.Row():
        brief_output = gr.Markdown(label="Audit Brief")
    with gr.Row():
        individual_output = gr.Markdown(label="Specialist Audit Reports")
    with gr.Row():
        final_output = gr.Markdown(label="Executive Summary")
    gr.Examples(
        examples=[
            ["Azure Kubernetes cluster in production with public API endpoint, admin service accounts with wildcard permissions, no MFA enforced, direct SSH access to nodes, no encryption at rest"],
            ["AWS EC2 instances in DEV environment, public S3 buckets, root account used for daily operations, no CloudTrail logging enabled, security groups with 0.0.0.0/0 inbound rules"],
            ["GCP Cloud Run services with public endpoints, service accounts with Owner role, no VPC service controls, customer data stored without encryption, no audit logging"],
        ],
        inputs=infra_input,
        label="Try these examples"
    )
    audit_btn.click(
        fn=run_security_audit,
        inputs=infra_input,
        outputs=[brief_output, individual_output, final_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        auth=[("bala", "securityaudit2026")]
    )