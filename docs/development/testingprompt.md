## **CONTEXT & PROBLEM STATEMENT**
You are conducting a comprehensive review to ensure that implementation changes align with intended functionality and requirements, rather than simply modifying code to pass tests. This addresses the critical anti-pattern of "test-passing development" versus genuine "test-driven development."

## **PHASE 1: REQUIREMENT VERIFICATION & UNDERSTANDING**

**Step 1.1: Requirements Discovery**
```bash
# Locate and analyze core requirements
find . -name "project_overview.md" -o -name "requirements*.md" -o -name "README.md" | head -5
grep -r "Phase 1" docs/ --include="*.md" -n
```

**Step 1.2: Requirement Validation Checklist**
Before proceeding, verify you can answer these questions with specific evidence:
- [ ] **Functional Requirements**: What specific behavior should the system exhibit?
- [ ] **Acceptance Criteria**: What are the measurable success criteria?
- [ ] **User Stories**: What user needs are being addressed?
- [ ] **Technical Constraints**: What are the system limitations and requirements?
- [ ] **Integration Points**: How should this component interact with other parts?

**Step 1.3: Clarification Protocol**
If ANY requirement is ambiguous or unclear:
1. **STOP** implementation work
2. Document specific questions: "Requirement X is unclear because..."
3. Request clarification with specific examples
4. Do not proceed until clarity is achieved

## **PHASE 2: TEST VALIDATION PROTOCOL**

**Step 2.1: Test Quality Assessment**
```bash
# Analyze test structure and quality
pytest --collect-only -q  # List all tests
pytest --tb=short -v     # Run tests with detailed output
```

**Step 2.2: Test Correctness Verification**
For each test, verify:
- [ ] **Test Purpose**: Does the test validate intended behavior (not just current implementation)?
- [ ] **Assertion Quality**: Uses proper `assert` statements with meaningful error messages
- [ ] **Test Data**: Uses realistic data, not hardcoded values that mask real issues
- [ ] **Edge Cases**: Tests boundary conditions and error scenarios
- [ ] **Independence**: Tests don't depend on external state or other tests

**Step 2.3: Anti-Pattern Detection**
Look for these TEST ANTI-PATTERNS:
```python
# ❌ BAD: Hardcoded values that mask real logic
def test_calculation():
    result = calculate_fee(user_type="premium")
    assert result == 42  # Magic number - why 42?

# ❌ BAD: Testing implementation details instead of behavior
def test_internal_method():
    obj = MyClass()
    assert obj._private_method() == "expected"  # Testing internals

# ❌ BAD: Tests that return values instead of using assertions
def test_validation():
    return validate_input("test") == True  # Should use assert

# ✅ GOOD: Tests that validate intended behavior
def test_premium_user_gets_discount():
    user = create_premium_user()
    fee = calculate_fee(user)
    assert fee == PREMIUM_DISCOUNT_RATE * BASE_FEE, f"Expected premium discount, got {fee}"
```

## **PHASE 3: IMPLEMENTATION ALIGNMENT VERIFICATION**

**Step 3.1: Requirements-Code Alignment Check**
```bash
# Search for requirement keywords in code
grep -r "TODO\|FIXME\|XXX" src/ --include="*.py" -n
grep -r "# test" src/ --include="*.py" -n  # Look for test-specific hacks
```

**Step 3.2: Implementation Anti-Pattern Detection**
Look for these CODE ANTI-PATTERNS:
- **Hardcoded Test Values**: Constants that exist only to satisfy tests
- **Conditional Test Logic**: `if testing:` or similar branches
- **Mock Overuse**: Excessive mocking that masks real integration issues
- **Test-Specific Methods**: Methods that serve no business purpose
- **Incomplete Error Handling**: Try-catch blocks that hide real errors

**Step 3.3: Behavioral Verification**
```bash
# Test actual behavior vs intended behavior
pytest -v --tb=long  # Get full error traces
pytest --pdb-trace   # Drop into debugger on failures
```

## **PHASE 4: EVIDENCE DOCUMENTATION**

**Step 4.1: Evidence Requirements**
For each finding, document:
```
FINDING: [Specific issue found]
EVIDENCE: [File path:line number with actual code snippet]
REQUIREMENT: [Which requirement this relates to]
IMPACT: [How this affects intended functionality]
CONFIDENCE: [HIGH/MEDIUM/LOW based on evidence quality]
```

**Step 4.2: Test Verification Results**
Document in `testverification.md`:
```markdown
## Test Verification Results

### Requirements Alignment
- [ ] All requirements clearly understood
- [ ] Acceptance criteria defined and measurable
- [ ] Edge cases and error conditions identified

### Test Quality Assessment
- [ ] Tests validate behavior, not implementation
- [ ] Meaningful assertions with clear error messages
- [ ] Realistic test data and scenarios
- [ ] Independent, repeatable tests

### Implementation Verification
- [ ] Code implements intended functionality
- [ ] No test-specific workarounds or hacks
- [ ] Proper error handling and edge case management
- [ ] Integration points work as specified

### Issues Found
[List specific issues with evidence]

### Corrective Actions Needed
[List specific actions to align with requirements]
```

## **PHASE 5: CORRECTIVE ACTION PROTOCOL**

**Step 5.1: Prioritization Matrix**
- **Critical**: Functional requirements not met
- **High**: Tests don't validate intended behavior
- **Medium**: Code quality issues that affect maintainability
- **Low**: Minor deviations from best practices

**Step 5.2: Correction Approach**
1. **Fix Requirements First**: Ensure clear understanding before code changes
2. **Fix Tests Second**: Ensure tests validate intended behavior
3. **Fix Implementation Last**: Align code with requirements and tests
4. **Validate Integration**: Ensure all components work together as intended

**Step 5.3: Re-verification**
After corrections:
```bash
# Re-run full verification
pytest -v --tb=short
# Manual behavior testing
# Requirements re-check
```

## **SUCCESS CRITERIA**
- All requirements clearly understood and documented
- Tests validate intended behavior, not just current implementation
- Implementation fulfills all functional requirements
- No test-specific workarounds or hardcoded values
- All evidence documented with specific file references
- Integration points work as specified

## **TOOLS TO USE**
- `pytest --collect-only` - Understand test structure
- `pytest -v --tb=long` - Get detailed test results
- `grep -r "pattern" src/` - Search for anti-patterns
- Code review for hardcoded values and test-specific logic

This protocol ensures that implementation truly serves the intended functionality rather than merely satisfying potentially flawed tests. 