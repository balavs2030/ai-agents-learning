import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent
from fastmcp import Client

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────
# Define LangChain tools that call MCP server
# ─────────────────────────────────────────

@tool
def audit_iam_policy(cloud_provider: str, policy_description: str) -> str:
    """Audits IAM policies for security risks and least privilege violations.
    Args:
        cloud_provider: The cloud platform (Azure, AWS, or GCP)
        policy_description: Description of the IAM policy to audit
    """
    async def _call():
        async with Client(Path("security_server.py")) as c:
            result = await c.call_tool(
                "audit_iam_policy",
                arguments={
                    "cloud_provider": cloud_provider,
                    "policy_description": policy_description
                }
            )
            return result.content[0].text if result.content else "No result"
    return asyncio.run(_call())

@tool
def check_network_security(infrastructure_description: str, cloud_provider: str) -> str:
    """Checks network security configuration for vulnerabilities.
    Args:
        infrastructure_description: Description of the network infrastructure
        cloud_provider: The cloud platform (Azure, AWS, or GCP)
    """
    async def _call():
        async with Client(Path("security_server.py")) as c:
            result = await c.call_tool(
                "check_network_security",
                arguments={
                    "infrastructure_description": infrastructure_description,
                    "cloud_provider": cloud_provider
                }
            )
            return result.content[0].text if result.content else "No result"
    return asyncio.run(_call())

@tool
def generate_compliance_report(
    infrastructure_description: str,
    compliance_framework: str,
    cloud_provider: str
) -> str:
    """Generates a compliance assessment report against security frameworks.
    Args:
        infrastructure_description: Description of the infrastructure
        compliance_framework: Target framework (CIS, NIST, SOC2, PCI-DSS, HIPAA)
        cloud_provider: The cloud platform (Azure, AWS, or GCP)
    """
    async def _call():
        async with Client(Path("security_server.py")) as c:
            result = await c.call_tool(
                "generate_compliance_report",
                arguments={
                    "infrastructure_description": infrastructure_description,
                    "compliance_framework": compliance_framework,
                    "cloud_provider": cloud_provider
                }
            )
            return result.content[0].text if result.content else "No result"
    return asyncio.run(_call())

# ─────────────────────────────────────────
# Run the security audit
# ─────────────────────────────────────────

async def run_security_audit(infrastructure: str):
    print("\nStarting Security Audit...")
    print(f"Infrastructure: {infrastructure}")
    print("="*60)

    tools = [audit_iam_policy, check_network_security, generate_compliance_report]
    print(f"Tools loaded: {[t.name for t in tools]}")

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0
    )

    agent = create_agent(model=llm, tools=tools)

    response = await agent.ainvoke({
        "messages": [
            {
                "role": "system",
                "content": """You are a senior cloud security auditor with 20 years
                of experience across Azure, AWS and GCP. When given an infrastructure 
                description you MUST use all three tools:
                1. audit_iam_policy - audit IAM and access management
                2. check_network_security - check network configuration
                3. generate_compliance_report - generate NIST compliance report
                After using all three tools provide a comprehensive final summary."""
            },
            {
                "role": "user",
                "content": f"""Conduct a full security audit for:

{infrastructure}

Use all three security tools and provide a comprehensive assessment."""
            }
        ]
    })

    print("\n" + "="*60)
    print("FINAL SECURITY AUDIT REPORT")
    print("="*60)
    print(response["messages"][-1].content)

async def main():
    infrastructure = """Azure Kubernetes cluster in production with:
    - Public API server endpoint enabled
    - Admin service accounts with wildcard permissions
    - No MFA enforced for developers
    - Direct SSH access to nodes allowed
    - No encryption mentioned for storage"""

    await run_security_audit(infrastructure)

if __name__ == "__main__":
    asyncio.run(main())