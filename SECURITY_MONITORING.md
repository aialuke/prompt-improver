# Security Vulnerability Monitoring

## Active Security Issues

### PyTorch CVE-2025-3730 (GHSA-887c-mr87-cxwp)
**Status**: MONITORING - No fix available yet  
**Priority**: HIGH  
**Impact**: DoS via ctc_loss function manipulation  
**Current Version**: 2.7.1  
**Date Added**: 2025-07-23  

#### Description
PyTorch vulnerability CVE-2025-3730 allows denial of service attacks through manipulation of the ctc_loss function. This affects our ML pipeline components that use PyTorch for model training and inference.

#### Affected Components
- ML model training pipelines
- Inference engines using PyTorch
- Any code using `torch.nn.functional.ctc_loss`

#### Monitoring Actions
1. **Weekly Check**: Monitor PyTorch security advisories at https://github.com/pytorch/pytorch/security/advisories
2. **Version Tracking**: Check for new PyTorch releases that address CVE-2025-3730
3. **Workaround**: Avoid using ctc_loss function in production until patch is available
4. **Alternative**: Consider using alternative loss functions where possible

#### Next Review Date
**2025-07-30** (Weekly review)

#### Escalation Criteria
- If exploit code becomes publicly available
- If PyTorch releases a security patch
- If we detect any suspicious activity related to ctc_loss usage

---

## Resolved Security Issues

### Starlette CVE-2025-54121 (GHSA-2c2j-9gv5-cj73)
**Status**: RESOLVED  
**Priority**: CRITICAL  
**Impact**: DoS via large file uploads blocking main thread  
**Resolution**: Upgraded to Starlette 0.47.2+  
**Date Resolved**: 2025-07-23  

#### Resolution Details
- Upgraded from Starlette 0.46.2 to 0.47.2
- Updated requirements.lock to reflect new version
- Verified no breaking changes in application

---

## Security Monitoring Process

### Weekly Security Review
1. Check all dependency security advisories
2. Review PyTorch CVE-2025-3730 status
3. Scan for new vulnerabilities in dependencies
4. Update this document with findings

### Monthly Security Audit
1. Run comprehensive security scan
2. Review hardcoded secrets detection
3. Validate all environment variable usage
4. Update security documentation

### Automated Monitoring
- GitHub Dependabot alerts enabled
- Security scanning in CI/CD pipeline
- Dependency vulnerability checks on pull requests

---

## Contact Information
**Security Team**: security@apes.local  
**Escalation**: CTO for critical vulnerabilities  
**Documentation**: This file (SECURITY_MONITORING.md)  

Last Updated: 2025-07-23
