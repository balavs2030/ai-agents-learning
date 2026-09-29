from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    name="Cloud Security Auditor",
    instructions="""You are a cloud security audit server with expertise in 
    Azure, AWS and GCP security. You provide security assessments, IAM audits,
    network security checks, and compliance reports."""
)

# ─────────────────────────────────────────
# TOOL 1 — IAM Policy Auditor
# ─────────────────────────────────────────
@mcp.tool()
def audit_iam_policy(
    cloud_provider: str,
    policy_description: str
) -> str:
    """
    Audits IAM policies for security risks and least privilege violations.
    
    Args:
        cloud_provider: The cloud platform (Azure, AWS, or GCP)
        policy_description: Description of the IAM policy to audit
    
    Returns:
        Detailed IAM security assessment with findings and recommendations
    """
    findings = []
    risk_level = "LOW"
    
    policy_lower = policy_description.lower()
    
    # Check for common IAM risks
    if any(term in policy_lower for term in ["*", "all", "wildcard", "admin"]):
        findings.append("CRITICAL: Wildcard or admin permissions detected — violates least privilege principle")
        risk_level = "CRITICAL"
    
    if "mfa" not in policy_lower and "multi-factor" not in policy_lower:
        findings.append("HIGH: MFA not mentioned — enforce MFA for all privileged access")
        if risk_level not in ["CRITICAL"]:
            risk_level = "HIGH"
    
    if "rotation" not in policy_lower and "rotate" not in policy_lower:
        findings.append("MEDIUM: No key rotation policy detected — implement 90-day rotation")
        if risk_level not in ["CRITICAL", "HIGH"]:
            risk_level = "MEDIUM"
    
    if cloud_provider.lower() == "azure":
        findings.append("Azure: Verify Azure AD PIM is enabled for just-in-time privileged access")
        findings.append("Azure: Ensure Conditional Access policies are applied to all admin accounts")
    elif cloud_provider.lower() == "aws":
        findings.append("AWS: Verify IAM Access Analyzer is enabled for external access detection")
        findings.append("AWS: Check for unused IAM roles older than 90 days")
    elif cloud_provider.lower() == "gcp":
        findings.append("GCP: Verify Workload Identity Federation replaces service account keys")
        findings.append("GCP: Check for basic roles (Owner/Editor) — replace with predefined roles")
    
    if not findings:
        findings.append("No immediate critical findings — policy appears to follow basic security principles")
    
    report = f"""
═══════════════════════════════════════
IAM SECURITY AUDIT REPORT
Cloud Provider: {cloud_provider.upper()}
Overall Risk Level: {risk_level}
═══════════════════════════════════════

FINDINGS:
{chr(10).join(f'  {i+1}. {f}' for i, f in enumerate(findings))}

RECOMMENDATIONS:
  • Implement least privilege — grant only permissions required for the specific task
  • Enable MFA for all human identities accessing cloud resources
  • Rotate all credentials and keys every 90 days
  • Review and remove unused permissions quarterly
  • Enable audit logging for all IAM changes

CIS BENCHMARK REFERENCE:
  • CIS {cloud_provider.upper()} Foundations Benchmark — Section 1 (Identity and Access Management)
"""
    return report


# ─────────────────────────────────────────
# TOOL 2 — Network Security Checker
# ─────────────────────────────────────────
@mcp.tool()
def check_network_security(
    infrastructure_description: str,
    cloud_provider: str
) -> str:
    """
    Checks network security configuration for vulnerabilities and misconfigurations.
    
    Args:
        infrastructure_description: Description of the network infrastructure
        cloud_provider: The cloud platform (Azure, AWS, or GCP)
    
    Returns:
        Network security assessment with specific findings and remediation steps
    """
    findings = []
    risk_level = "LOW"
    
    desc_lower = infrastructure_description.lower()
    
    # Check for common network risks
    if "public" in desc_lower and "private" not in desc_lower:
        findings.append("CRITICAL: Resources appear publicly exposed — implement private subnets")
        risk_level = "CRITICAL"
    
    if "0.0.0.0" in desc_lower or "any" in desc_lower:
        findings.append("CRITICAL: Open inbound rules detected (0.0.0.0/0) — restrict to known IP ranges")
        risk_level = "CRITICAL"
    
    if "ssh" in desc_lower and "bastion" not in desc_lower:
        findings.append("HIGH: Direct SSH access detected — use bastion host or VPN instead")
        if risk_level not in ["CRITICAL"]:
            risk_level = "HIGH"
    
    if "encryption" not in desc_lower and "tls" not in desc_lower and "https" not in desc_lower:
        findings.append("HIGH: No encryption in transit mentioned — enforce TLS 1.2+ for all traffic")
        if risk_level not in ["CRITICAL"]:
            risk_level = "HIGH"
    
    if "firewall" not in desc_lower and "nsg" not in desc_lower and "security group" not in desc_lower:
        findings.append("MEDIUM: No firewall rules mentioned — implement network segmentation")
        if risk_level not in ["CRITICAL", "HIGH"]:
            risk_level = "MEDIUM"
    
    # Cloud-specific checks
    if cloud_provider.lower() == "azure":
        findings.append("Azure: Enable Azure DDoS Protection Standard on VNets")
        findings.append("Azure: Use NSG Flow Logs for network traffic analysis")
    elif cloud_provider.lower() == "aws":
        findings.append("AWS: Enable VPC Flow Logs for all VPCs")
        findings.append("AWS: Use AWS Shield Standard for DDoS protection")
    elif cloud_provider.lower() == "gcp":
        findings.append("GCP: Enable VPC Flow Logs and Firewall Rules Logging")
        findings.append("GCP: Use Cloud Armor for DDoS and WAF protection")
    
    if not findings:
        findings.append("Network configuration appears to follow security best practices")
    
    report = f"""
═══════════════════════════════════════
NETWORK SECURITY ASSESSMENT
Cloud Provider: {cloud_provider.upper()}
Overall Risk Level: {risk_level}
═══════════════════════════════════════

FINDINGS:
{chr(10).join(f'  {i+1}. {f}' for i, f in enumerate(findings))}

RECOMMENDATIONS:
  • Implement network segmentation with private/public subnet separation
  • Restrict all inbound rules to specific IP ranges — no 0.0.0.0/0
  • Use bastion hosts or VPN for administrative access
  • Enforce TLS 1.2+ for all data in transit
  • Enable DDoS protection at the network perimeter
  • Implement Web Application Firewall (WAF) for public-facing endpoints

CIS BENCHMARK REFERENCE:
  • CIS {cloud_provider.upper()} Foundations Benchmark — Section 3 (Networking)
"""
    return report


# ─────────────────────────────────────────
# TOOL 3 — Compliance Report Generator
# ─────────────────────────────────────────
@mcp.tool()
def generate_compliance_report(
    infrastructure_description: str,
    compliance_framework: str,
    cloud_provider: str
) -> str:
    """
    Generates a compliance assessment report against major security frameworks.
    
    Args:
        infrastructure_description: Description of the infrastructure to assess
        compliance_framework: Target framework (CIS, NIST, SOC2, PCI-DSS, HIPAA)
        cloud_provider: The cloud platform (Azure, AWS, or GCP)
    
    Returns:
        Compliance gap analysis with specific control mappings
    """
    framework_controls = {
        "CIS": [
            "1.1 — Maintain inventory of cloud accounts",
            "1.2 — Ensure MFA is enabled for all accounts",
            "2.1 — Ensure CloudTrail/Audit Logs are enabled",
            "3.1 — Ensure no unrestricted inbound access",
            "4.1 — Ensure encryption at rest is enabled",
        ],
        "NIST": [
            "ID.AM — Asset Management: Maintain asset inventory",
            "PR.AC — Access Control: Implement least privilege",
            "PR.DS — Data Security: Encrypt data at rest and in transit",
            "DE.CM — Security Monitoring: Implement continuous monitoring",
            "RS.RP — Incident Response: Maintain response plan",
        ],
        "SOC2": [
            "CC6.1 — Logical and Physical Access Controls",
            "CC6.6 — Restrict access to authorized users",
            "CC7.1 — System Monitoring and Detection",
            "CC8.1 — Change Management Controls",
            "CC9.1 — Risk Mitigation Controls",
        ],
        "PCI-DSS": [
            "Req 1 — Install and maintain network security controls",
            "Req 2 — Apply secure configurations to all components",
            "Req 3 — Protect stored account data",
            "Req 7 — Restrict access by business need to know",
            "Req 10 — Log and monitor all access to system components",
        ],
        "HIPAA": [
            "164.312(a) — Access Control",
            "164.312(b) — Audit Controls",
            "164.312(c) — Integrity Controls",
            "164.312(d) — Person Authentication",
            "164.312(e) — Transmission Security",
        ]
    }
    
    framework_upper = compliance_framework.upper()
    controls = framework_controls.get(
        framework_upper,
        framework_controls["NIST"]
    )
    
    desc_lower = infrastructure_description.lower()
    
    # Assess each control
    control_status = []
    gaps = []
    passed = 0
    
    for control in controls:
        # Simple heuristic assessment
        if any(term in desc_lower for term in ["encrypt", "mfa", "log", "monitor", "access control"]):
            control_status.append(f" {control}")
            passed += 1
        else:
            control_status.append(f" {control} — GAP IDENTIFIED")
            gaps.append(control)
    
    compliance_score = int((passed / len(controls)) * 100)
    
    report = f"""
═══════════════════════════════════════
COMPLIANCE ASSESSMENT REPORT
Framework: {framework_upper}
Cloud Provider: {cloud_provider.upper()}
Compliance Score: {compliance_score}%
═══════════════════════════════════════

CONTROL STATUS:
{chr(10).join(control_status)}

IDENTIFIED GAPS ({len(gaps)}):
{chr(10).join(f' {g}' for g in gaps) if gaps else '  None identified'}

REMEDIATION PRIORITY:
  1. Address all CRITICAL gaps within 24 hours
  2. HIGH severity gaps within 72 hours  
  3. MEDIUM severity gaps within 30 days
  4. Schedule quarterly compliance reviews

NEXT STEPS:
  • Engage compliance team for formal assessment
  • Document all controls and evidence
  • Implement continuous compliance monitoring
  • Schedule penetration testing
  • Review and update security policies annually

OVERALL STATUS: {'COMPLIANT' if compliance_score >= 80 else 'PARTIALLY COMPLIANT' if compliance_score >= 60 else 'NON-COMPLIANT'}
"""
    return report


# ─────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()