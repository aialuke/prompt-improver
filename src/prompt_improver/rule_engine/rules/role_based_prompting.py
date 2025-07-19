"""RoleBasedPromptingRule: Enhanced rule for expert role assignment and persona consistency.

Based on research synthesis from:
- Anthropic role-based prompting best practices
- Expert persona research and effectiveness studies
- Domain-specific expertise assignment patterns
- Persona consistency maintenance techniques
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ..base import (
    BasePromptRule,
    RuleCheckResult,
    TransformationResult,
)

# Domain detection patterns
DOMAIN_PATTERNS = {
    "technical": [
        r"\b(code|coding|programming|software|algorithm|system|API|framework|database)\b",
        r"\b(technical|technology|engineering|development|architecture)\b",
        r"\b(debug|optimize|implement|deploy|configure)\b",
    ],
    "business": [
        r"\b(business|marketing|sales|finance|revenue|profit|strategy|market)\b",
        r"\b(customer|client|stakeholder|investor|budget|ROI)\b",
        r"\b(corporate|company|organization|enterprise)\b",
    ],
    "academic": [
        r"\b(research|study|academic|scholarly|paper|thesis|methodology)\b",
        r"\b(hypothesis|analysis|findings|conclusion|literature|peer)\b",
        r"\b(university|professor|student|curriculum)\b",
    ],
    "creative": [
        r"\b(creative|story|narrative|poem|art|design|writing|content)\b",
        r"\b(imagination|inspiration|artistic|aesthetic|style)\b",
        r"\b(author|writer|artist|designer|creator)\b",
    ],
    "legal": [
        r"\b(legal|law|contract|regulation|compliance|policy|statute)\b",
        r"\b(attorney|lawyer|counsel|court|judge|litigation)\b",
        r"\b(rights|obligations|liability|jurisdiction)\b",
    ],
    "medical": [
        r"\b(medical|health|patient|diagnosis|treatment|clinical|therapeutic)\b",
        r"\b(doctor|physician|nurse|healthcare|hospital|clinic)\b",
        r"\b(symptoms|medicine|therapy|surgical|pharmaceutical)\b",
    ],
    "education": [
        r"\b(teach|learn|education|training|course|curriculum|instruction)\b",
        r"\b(teacher|instructor|student|learner|pedagogy)\b",
        r"\b(classroom|lesson|assessment|evaluation)\b",
    ],
    "financial": [
        r"\b(financial|investment|banking|accounting|audit|tax)\b",
        r"\b(money|capital|assets|portfolio|risk|returns)\b",
        r"\b(analyst|advisor|accountant|banker)\b",
    ],
    "scientific": [
        r"\b(scientific|science|experiment|research|hypothesis|theory)\b",
        r"\b(data|analysis|methodology|results|conclusion)\b",
        r"\b(scientist|researcher|laboratory|study)\b",
    ],
}

# Expertise levels and their characteristics
EXPERTISE_LEVELS = {
    "junior": {
        "experience": "1-3 years",
        "characteristics": [
            "eager to learn",
            "follows established patterns",
            "seeks guidance",
        ],
        "tone": "enthusiastic and learning-oriented",
    },
    "mid_level": {
        "experience": "3-7 years",
        "characteristics": [
            "solid foundation",
            "practical experience",
            "problem-solving focus",
        ],
        "tone": "confident and practical",
    },
    "senior_level": {
        "experience": "7-15 years",
        "characteristics": [
            "deep expertise",
            "strategic thinking",
            "mentorship capability",
        ],
        "tone": "authoritative and insightful",
    },
    "expert": {
        "experience": "15+ years",
        "characteristics": [
            "industry leader",
            "innovative thinking",
            "comprehensive knowledge",
        ],
        "tone": "visionary and comprehensive",
    },
}

# Expert personas by domain
EXPERT_PERSONAS = {
    "technical": {
        "senior_level": "a Senior Software Architect with 10+ years of experience in system design, best practices, and emerging technologies",
        "expert": "a Distinguished Engineer and technology leader with 15+ years of experience spanning multiple programming languages, architectural patterns, and industry innovations",
        "mid_level": "an experienced Software Developer with 5+ years of hands-on coding and system implementation experience",
        "junior": "a Junior Developer with strong fundamentals and enthusiasm for learning new technologies",
    },
    "business": {
        "senior_level": "a Senior Business Strategist with 12+ years of experience in market analysis, strategic planning, and organizational growth",
        "expert": "a Chief Strategy Officer with 20+ years of experience in business transformation, market leadership, and executive decision-making",
        "mid_level": "a Business Analyst with 6+ years of experience in process optimization and strategic planning",
        "junior": "a Business Associate with strong analytical skills and eagerness to understand market dynamics",
    },
    "academic": {
        "senior_level": "a Professor with 15+ years of research experience and numerous published papers in the field",
        "expert": "a Distinguished Professor and department head with 25+ years of groundbreaking research and academic leadership",
        "mid_level": "an Associate Professor with 8+ years of research experience and established expertise",
        "junior": "a PhD candidate with strong research foundations and current knowledge of recent developments",
    },
    "creative": {
        "senior_level": "a Creative Director with 12+ years of experience in brand development, storytelling, and creative campaign leadership",
        "expert": "an Award-winning Creative Director with 20+ years of experience in innovative design, brand strategy, and creative vision",
        "mid_level": "a Senior Creative with 7+ years of experience in content creation and design implementation",
        "junior": "a Creative Professional with fresh perspectives and enthusiasm for innovative approaches",
    },
    "legal": {
        "senior_level": "a Senior Partner with 15+ years of legal practice and extensive experience in regulatory compliance and contract law",
        "expert": "a Managing Partner and legal expert with 25+ years of experience in complex litigation, regulatory affairs, and legal strategy",
        "mid_level": "a Senior Associate with 8+ years of legal practice and specialized expertise",
        "junior": "a Junior Associate with strong legal foundations and attention to detail",
    },
    "medical": {
        "senior_level": "a Senior Physician with 15+ years of clinical experience and specialized expertise in diagnosis and treatment",
        "expert": "a Department Head and Medical Expert with 25+ years of clinical practice, research, and medical innovation",
        "mid_level": "an Attending Physician with 8+ years of clinical experience and specialized training",
        "junior": "a Medical Resident with strong medical knowledge and commitment to patient care",
    },
    "education": {
        "senior_level": "a Senior Educator with 15+ years of teaching experience and curriculum development expertise",
        "expert": "an Education Director with 25+ years of experience in pedagogical innovation and educational leadership",
        "mid_level": "an experienced Teacher with 8+ years of classroom experience and instructional expertise",
        "junior": "a New Educator with strong pedagogical foundations and enthusiasm for student success",
    },
    "financial": {
        "senior_level": "a Senior Financial Advisor with 12+ years of experience in investment strategy and portfolio management",
        "expert": "a Chief Financial Officer with 20+ years of experience in financial planning, risk management, and strategic finance",
        "mid_level": "a Financial Analyst with 6+ years of experience in financial modeling and analysis",
        "junior": "a Junior Financial Analyst with strong quantitative skills and eagerness to learn market dynamics",
    },
    "scientific": {
        "senior_level": "a Senior Research Scientist with 15+ years of experience in experimental design and scientific methodology",
        "expert": "a Principal Scientist and research leader with 25+ years of groundbreaking research and scientific innovation",
        "mid_level": "a Research Scientist with 8+ years of laboratory experience and published research",
        "junior": "a Research Associate with strong scientific foundations and enthusiasm for discovery",
    },
}


class RoleBasedPromptingRule(BasePromptRule):
    """Enhanced role-based prompting rule using Anthropic research patterns.

    Features:
    - Automatic domain detection from prompt content
    - Expert persona assignment based on task complexity
    - Persona consistency maintenance throughout interaction
    - Configurable expertise depth and credentials
    """

    def __init__(self):
        # Research-validated default parameters
        self.config = {
            "auto_detect_domain": True,
            "use_system_prompts": True,
            "maintain_persona_consistency": True,
            "expertise_depth": "senior_level",
            "include_credentials": True,
            "domain_knowledge_depth": "expert",
            "persona_voice_consistency": True,
        }

        # Attributes for dynamic loading system
        self.rule_id = "role_based_prompting"
        self.priority = 6

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {
            "name": "Expert Role Assignment Rule",
            "type": "Context",
            "description": "Assigns appropriate expert personas based on Anthropic best practices for role-based prompting",
            "category": "context",
            "research_foundation": [
                "Anthropic role-based prompting",
                "Expert persona research",
                "Domain-specific expertise studies",
            ],
            "version": "2.0.0",
            "priority": self.priority,
            "source": "Research Synthesis 2025",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Check if prompt would benefit from expert role assignment"""
        role_metrics = self._analyze_role_requirements(prompt)

        # Determine if role assignment should be applied
        applies = (
            role_metrics["domain"] != "general"
            and not role_metrics["already_has_role"]
            and role_metrics["benefits_from_expertise"]
        )

        confidence = 0.9 if applies else 0.85

        return RuleCheckResult(
            applies=applies,
            confidence=confidence,
            metadata={
                "domain": role_metrics["domain"],
                "task_complexity": role_metrics["task_complexity"],
                "expertise_needed": role_metrics["expertise_needed"],
                "already_has_role": role_metrics["already_has_role"],
                "benefits_from_expertise": role_metrics["benefits_from_expertise"],
                "recommended_persona": role_metrics["recommended_persona"],
                "credential_requirements": role_metrics["credential_requirements"],
            },
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply expert role assignment enhancement"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True, improved_prompt=prompt, confidence=1.0, transformations=[]
            )

        role_metrics = check_result.metadata
        improved_prompt = prompt
        transformations = []

        # Get domain and expertise level
        domain = role_metrics.get("domain", "general")
        expertise_level = self.config["expertise_depth"]

        # Generate expert persona
        persona = self._get_expert_persona(domain, expertise_level)

        if persona:
            # Apply role assignment with appropriate formatting
            improved_prompt, role_transformations = self._apply_role_assignment(
                improved_prompt, persona, domain, role_metrics
            )
            transformations.extend(role_transformations)

        # Calculate confidence based on domain match and persona quality
        confidence = min(
            0.95, 0.75 + (0.05 if domain != "general" else 0) + (0.15 if persona else 0)
        )

        return TransformationResult(
            success=True,
            improved_prompt=improved_prompt,
            confidence=confidence,
            transformations=transformations,
        )

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction for role assignment"""
        return """
<instruction>
Apply expert role assignment using Anthropic research patterns:

1. DOMAIN DETECTION:
   - Identify the primary subject area and expertise required
   - Match task complexity with appropriate expertise level
   - Consider specialized knowledge requirements

2. PERSONA ASSIGNMENT:
   - Assign specific expert persona with relevant credentials
   - Include years of experience and specialization areas
   - Maintain consistency with domain requirements

3. PERSONA CONSISTENCY:
   - Use expert voice and terminology throughout response
   - Apply domain-specific knowledge and insights
   - Maintain professional tone appropriate to expertise level

4. SYSTEM PROMPT INTEGRATION:
   - Begin with clear role statement: "You are [expert persona]"
   - Include relevant credentials and experience
   - Set expectation for expert-level response quality

Focus on authentic expertise that enhances response quality and credibility.
</instruction>
"""

    def _analyze_role_requirements(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt to determine expert role requirements"""
        # Detect domain using pattern matching
        domain = (
            self._detect_domain(prompt)
            if self.config["auto_detect_domain"]
            else "general"
        )

        # Check if prompt already has role assignment
        role_indicators = [
            "you are",
            "act as",
            "as a",
            "as an",
            "role of",
            "perspective of",
            "expert",
            "professional",
            "specialist",
            "consultant",
        ]
        already_has_role = any(
            indicator in prompt.lower() for indicator in role_indicators
        )

        # Determine task complexity
        task_complexity = self._assess_task_complexity(prompt)

        # Determine if expertise would benefit this task
        expertise_beneficial_domains = [
            "technical",
            "business",
            "academic",
            "legal",
            "medical",
            "financial",
            "scientific",
            "creative",
        ]
        benefits_from_expertise = domain in expertise_beneficial_domains

        # Determine expertise level needed
        if task_complexity > 0.8:
            expertise_needed = "expert"
        elif task_complexity > 0.6:
            expertise_needed = "senior_level"
        elif task_complexity > 0.4:
            expertise_needed = "mid_level"
        else:
            expertise_needed = "junior"

        # Get recommended persona
        recommended_persona = self._get_expert_persona(domain, expertise_needed)

        # Determine credential requirements
        credential_requirements = self._assess_credential_requirements(prompt, domain)

        return {
            "domain": domain,
            "task_complexity": task_complexity,
            "expertise_needed": expertise_needed,
            "already_has_role": already_has_role,
            "benefits_from_expertise": benefits_from_expertise,
            "recommended_persona": recommended_persona,
            "credential_requirements": credential_requirements,
        }

    def _detect_domain(self, prompt: str) -> str:
        """Detect the primary domain from prompt content"""
        prompt_lower = prompt.lower()
        domain_scores = {}

        for domain, patterns in DOMAIN_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                score += matches
            domain_scores[domain] = score

        # Return domain with highest score, or 'general' if no strong matches
        if domain_scores:
            best_domain = max(domain_scores, key=lambda k: domain_scores.get(k, 0))
            if domain_scores[best_domain] > 0:
                return best_domain

        return "general"

    def _assess_task_complexity(self, prompt: str) -> float:
        """Assess the complexity of the task to determine expertise level needed"""
        words = prompt.lower().split()

        # Complexity indicators
        complex_indicators = [
            "complex",
            "advanced",
            "sophisticated",
            "comprehensive",
            "detailed",
            "strategic",
            "innovative",
            "optimize",
            "analyze",
            "evaluate",
            "design",
            "implement",
            "develop",
            "create",
            "solve",
        ]

        complexity_count = sum(1 for word in words if word in complex_indicators)

        # Task scope indicators
        scope_indicators = [
            "system",
            "framework",
            "architecture",
            "strategy",
            "methodology",
            "process",
            "workflow",
            "integration",
            "scalable",
            "enterprise",
        ]

        scope_count = sum(1 for word in words if word in scope_indicators)

        # Calculate complexity score
        base_complexity = complexity_count / max(len(words), 1) * 10
        scope_complexity = scope_count / max(len(words), 1) * 5

        # Length-based complexity (longer prompts often indicate more complex tasks)
        length_complexity = min(0.3, len(words) / 100)

        total_complexity = min(
            1.0, base_complexity + scope_complexity + length_complexity
        )

        return total_complexity

    def _get_expert_persona(self, domain: str, expertise_level: str) -> str | None:
        """Get appropriate expert persona for domain and expertise level"""
        if domain == "general" or domain not in EXPERT_PERSONAS:
            return None

        personas = EXPERT_PERSONAS.get(domain, {})
        persona = personas.get(expertise_level)

        # Fallback to senior_level if requested level not available
        if not persona and expertise_level != "senior_level":
            persona = personas.get("senior_level")

        return persona

    def _assess_credential_requirements(
        self, prompt: str, domain: str
    ) -> dict[str, Any]:
        """Assess what credentials or qualifications should be emphasized"""
        requirements = {
            "certifications": False,
            "education": False,
            "publications": False,
            "awards": False,
            "experience_years": True,  # Always include experience
        }

        prompt_lower = prompt.lower()

        # Check for specific credential indicators
        if any(
            word in prompt_lower for word in ["certified", "license", "qualification"]
        ):
            requirements["certifications"] = True

        if any(
            word in prompt_lower for word in ["academic", "research", "study", "paper"]
        ):
            requirements["publications"] = True
            requirements["education"] = True

        if any(
            word in prompt_lower for word in ["award", "recognition", "achievement"]
        ):
            requirements["awards"] = True

        # Domain-specific requirements
        if domain in ["academic", "scientific"]:
            requirements["education"] = True
            requirements["publications"] = True
        elif domain in ["legal", "medical"]:
            requirements["certifications"] = True
            requirements["education"] = True

        return requirements

    def _apply_role_assignment(
        self, prompt: str, persona: str, domain: str, metrics: dict
    ) -> tuple[str, list[dict]]:
        """Apply role assignment with appropriate formatting"""
        if not persona:
            return prompt, []

        transformations = []

        # Create role assignment based on configuration
        if self.config["use_system_prompts"]:
            # System prompt style (Anthropic recommended)
            role_assignment = f"You are {persona}."

            # Add credential emphasis if configured
            if self.config["include_credentials"]:
                credentials = self._generate_credential_emphasis(
                    domain, metrics.get("credential_requirements", {})
                )
                if credentials:
                    role_assignment += f" {credentials}"

            # Add consistency reminder
            if self.config["maintain_persona_consistency"]:
                consistency_note = " Please provide expert-level insights and maintain your professional perspective throughout your response."
                role_assignment += consistency_note

            enhanced_prompt = f"{role_assignment}\n\n{prompt}"
        else:
            # Inline role assignment
            role_assignment = f"As {persona}, please address the following:"
            enhanced_prompt = f"{role_assignment}\n\n{prompt}"

        transformations.append({
            "type": "role_based_prompting",
            "description": f"Assigned {domain} expert persona with {self.config['expertise_depth']} expertise level",
            "domain": domain,
            "expertise_level": self.config["expertise_depth"],
            "persona": persona,
            "research_basis": "Anthropic role-based prompting best practices",
        })

        return enhanced_prompt, transformations

    def _generate_credential_emphasis(
        self, domain: str, requirements: dict[str, Any]
    ) -> str:
        """Generate credential emphasis based on domain and requirements"""
        credential_parts = []

        if requirements.get("education") and domain in [
            "academic",
            "scientific",
            "medical",
            "legal",
        ]:
            education_emphasis = {
                "academic": "with a PhD in your field",
                "scientific": "with advanced degrees and research credentials",
                "medical": "with board certification and medical degree",
                "legal": "with a Juris Doctor and bar admission",
            }
            credential_parts.append(education_emphasis.get(domain, ""))

        if requirements.get("certifications"):
            credential_parts.append("holding relevant professional certifications")

        if requirements.get("publications") and domain in ["academic", "scientific"]:
            credential_parts.append("with numerous peer-reviewed publications")

        if requirements.get("awards"):
            credential_parts.append("recognized for excellence in your field")

        if credential_parts:
            # Clean up empty strings and join
            credential_parts = [part for part in credential_parts if part.strip()]
            if len(credential_parts) == 1:
                return credential_parts[0].capitalize()
            if len(credential_parts) == 2:
                return f"{credential_parts[0].capitalize()} and {credential_parts[1]}"
            return f"{', '.join(credential_parts[:-1]).capitalize()}, and {credential_parts[-1]}"

        return ""
