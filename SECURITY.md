# Security Policy

This software is verified for security for official releases and as such does
not make promises about the quality of the product for patches delivered between
releases.

## Security Boundaries

The ML SDK Scenario Runner is not a sandbox or process-security boundary. Input
scenarios, workload binaries, and resources are expected to come from trusted
sources. The Scenario Runner also relies on the operating system, filesystem,
Vulkan® Loader, and installable client driver (ICD) being trusted. Weaknesses
originating in those components cannot be addressed in the Scenario Runner.

## Reporting a Vulnerability

Security vulnerabilities may be reported to the Arm® Product Security Incident
Response Team (PSIRT) by sending an email to
[psirt@arm.com](mailto:psirt@arm.com).

For more information visit
<https://developer.arm.com/support/arm-security-updates/report-security-vulnerabilities>
