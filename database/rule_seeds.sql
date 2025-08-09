-- ===================================
-- Comprehensive Rule Metadata Seeding
-- Phase 1: Essential Database Container Isolation
-- ===================================
-- This file contains 185+ INSERT statements for rule metadata
-- to establish a comprehensive rule knowledge base

-- Clear existing rule_metadata to ensure clean seeding
TRUNCATE rule_metadata RESTART IDENTITY CASCADE;

-- ===================================
-- Core Clarity Enhancement Rules
-- ===================================

INSERT INTO rule_metadata (rule_id, rule_name, rule_category, rule_description, default_parameters, priority) VALUES
('clarity_basic', 'Basic Clarity Enhancement', 'core', 'Improves basic prompt clarity by removing ambiguous language', '{"min_improvement_threshold": 0.1, "ambiguity_detection": true}', 100),
('clarity_advanced', 'Advanced Clarity Enhancement', 'core', 'Advanced clarity improvement with context analysis', '{"context_analysis": true, "semantic_clarity": true}', 95),
('clarity_technical', 'Technical Clarity Rule', 'core', 'Enhances clarity for technical prompts and instructions', '{"technical_terms": true, "step_by_step": true}', 90),
('clarity_conversational', 'Conversational Clarity Rule', 'core', 'Improves clarity in conversational contexts', '{"natural_language": true, "tone_adjustment": true}', 85),
('clarity_academic', 'Academic Clarity Rule', 'core', 'Enhances clarity for academic and research contexts', '{"academic_tone": true, "precision_focus": true}', 88),
('clarity_business', 'Business Clarity Rule', 'core', 'Improves clarity for business communications', '{"professional_tone": true, "action_oriented": true}', 87),
('clarity_creative', 'Creative Clarity Rule', 'core', 'Balances clarity with creative expression', '{"preserve_creativity": true, "clear_intent": true}', 82),
('clarity_legal', 'Legal Clarity Rule', 'core', 'Enhances clarity while maintaining legal precision', '{"legal_precision": true, "unambiguous": true}', 92),
('clarity_medical', 'Medical Clarity Rule', 'core', 'Improves clarity for medical and healthcare contexts', '{"medical_accuracy": true, "patient_safety": true}', 94),
('clarity_educational', 'Educational Clarity Rule', 'core', 'Enhances clarity for educational content', '{"learning_objectives": true, "age_appropriate": true}', 86),

-- ===================================
-- Specificity Enhancement Rules
-- ===================================

('specificity_basic', 'Basic Specificity Enhancement', 'core', 'Makes prompts more specific and actionable', '{"context_expansion": true, "detail_level": "medium"}', 90),
('specificity_advanced', 'Advanced Specificity Rule', 'core', 'Advanced specificity with domain knowledge integration', '{"domain_context": true, "expert_level": true}', 85),
('specificity_quantitative', 'Quantitative Specificity Rule', 'core', 'Adds quantitative measures and metrics to prompts', '{"metrics_focus": true, "measurable_outcomes": true}', 88),
('specificity_temporal', 'Temporal Specificity Rule', 'core', 'Adds time-based specificity to prompts', '{"time_constraints": true, "deadline_awareness": true}', 83),
('specificity_geographic', 'Geographic Specificity Rule', 'core', 'Adds location-based specificity when relevant', '{"location_context": true, "regional_awareness": true}', 80),
('specificity_audience', 'Audience Specificity Rule', 'core', 'Tailors specificity to target audience', '{"audience_analysis": true, "demographic_awareness": true}', 86),
('specificity_outcome', 'Outcome Specificity Rule', 'core', 'Specifies desired outcomes and deliverables', '{"outcome_focus": true, "deliverable_clarity": true}', 89),
('specificity_constraint', 'Constraint Specificity Rule', 'core', 'Clearly defines constraints and limitations', '{"constraint_identification": true, "limitation_clarity": true}', 84),
('specificity_resource', 'Resource Specificity Rule', 'core', 'Specifies required resources and dependencies', '{"resource_mapping": true, "dependency_analysis": true}', 81),
('specificity_methodology', 'Methodology Specificity Rule', 'core', 'Specifies methods and approaches to be used', '{"methodology_clarity": true, "approach_definition": true}', 87),

-- ===================================
-- Structure Enhancement Rules
-- ===================================

('structure_basic', 'Basic Structure Enhancement', 'formatting', 'Improves basic prompt structure and organization', '{"add_formatting": true, "logical_flow": true}', 80),
('structure_advanced', 'Advanced Structure Rule', 'formatting', 'Advanced structural improvements with hierarchical organization', '{"hierarchical": true, "section_headers": true}', 75),
('structure_numbered', 'Numbered Structure Rule', 'formatting', 'Adds numbered lists and sequential organization', '{"numbered_lists": true, "sequential_flow": true}', 78),
('structure_bulleted', 'Bullet Point Structure Rule', 'formatting', 'Organizes content with bullet points and sub-points', '{"bullet_points": true, "nested_structure": true}', 76),
('structure_categorical', 'Categorical Structure Rule', 'formatting', 'Organizes content by categories and themes', '{"categorization": true, "thematic_grouping": true}', 77),
('structure_priority', 'Priority Structure Rule', 'formatting', 'Structures content by priority and importance', '{"priority_ordering": true, "importance_ranking": true}', 82),
('structure_chronological', 'Chronological Structure Rule', 'formatting', 'Organizes content in chronological order', '{"time_sequence": true, "chronological_flow": true}', 79),
('structure_process', 'Process Structure Rule', 'formatting', 'Structures content as step-by-step processes', '{"process_steps": true, "workflow_clarity": true}', 84),
('structure_comparison', 'Comparison Structure Rule', 'formatting', 'Structures content for easy comparison', '{"comparison_format": true, "parallel_structure": true}', 81),
('structure_decision', 'Decision Structure Rule', 'formatting', 'Structures content for decision-making processes', '{"decision_tree": true, "option_analysis": true}', 83),

-- ===================================
-- Context Enhancement Rules
-- ===================================

('context_basic', 'Basic Context Enhancement', 'content', 'Adds relevant context and background information', '{"context_depth": "medium", "background_info": true}', 70),
('context_advanced', 'Advanced Context Rule', 'content', 'Advanced contextual enhancement with stakeholder analysis', '{"stakeholder_analysis": true, "multi_perspective": true}', 65),
('context_historical', 'Historical Context Rule', 'content', 'Adds historical background and precedents', '{"historical_background": true, "precedent_analysis": true}', 68),
('context_industry', 'Industry Context Rule', 'content', 'Provides industry-specific context and knowledge', '{"industry_specific": true, "domain_expertise": true}', 72),
('context_competitive', 'Competitive Context Rule', 'content', 'Adds competitive landscape and market context', '{"competitive_analysis": true, "market_awareness": true}', 69),
('context_regulatory', 'Regulatory Context Rule', 'content', 'Includes regulatory and compliance context', '{"regulatory_awareness": true, "compliance_focus": true}', 74),
('context_cultural', 'Cultural Context Rule', 'content', 'Adds cultural and social context considerations', '{"cultural_sensitivity": true, "social_awareness": true}', 67),
('context_technological', 'Technological Context Rule', 'content', 'Includes relevant technological context', '{"tech_stack": true, "tool_awareness": true}', 71),
('context_economic', 'Economic Context Rule', 'content', 'Adds economic and financial context', '{"economic_factors": true, "cost_awareness": true}', 66),
('context_environmental', 'Environmental Context Rule', 'content', 'Includes environmental and sustainability context', '{"sustainability": true, "environmental_impact": true}', 63),

-- ===================================
-- Tone and Style Rules
-- ===================================

('tone_professional', 'Professional Tone Rule', 'style', 'Adjusts tone to professional standards', '{"formality_level": "high", "business_appropriate": true}', 75),
('tone_casual', 'Casual Tone Rule', 'style', 'Adjusts tone to casual, friendly communication', '{"formality_level": "low", "conversational": true}', 70),
('tone_authoritative', 'Authoritative Tone Rule', 'style', 'Establishes authoritative and expert tone', '{"expertise_demonstration": true, "confidence_level": "high"}', 78),
('tone_collaborative', 'Collaborative Tone Rule', 'style', 'Promotes collaborative and inclusive tone', '{"inclusive_language": true, "team_oriented": true}', 73),
('tone_persuasive', 'Persuasive Tone Rule', 'style', 'Enhances persuasive elements in communication', '{"persuasion_techniques": true, "compelling_language": true}', 76),
('tone_educational', 'Educational Tone Rule', 'style', 'Adopts educational and instructional tone', '{"teaching_approach": true, "learning_facilitation": true}', 74),
('tone_empathetic', 'Empathetic Tone Rule', 'style', 'Incorporates empathetic and understanding tone', '{"emotional_intelligence": true, "compassionate_language": true}', 72),
('tone_analytical', 'Analytical Tone Rule', 'style', 'Maintains analytical and objective tone', '{"objectivity": true, "data_driven": true}', 77),
('tone_inspirational', 'Inspirational Tone Rule', 'style', 'Adds inspirational and motivational elements', '{"motivation_focus": true, "uplifting_language": true}', 71),
('tone_diplomatic', 'Diplomatic Tone Rule', 'style', 'Ensures diplomatic and tactful communication', '{"tactful_language": true, "conflict_avoidance": true}', 79),

-- ===================================
-- Audience Adaptation Rules
-- ===================================

('audience_expert', 'Expert Audience Rule', 'audience', 'Adapts content for expert-level audiences', '{"technical_depth": "high", "assume_knowledge": true}', 85),
('audience_novice', 'Novice Audience Rule', 'audience', 'Adapts content for beginner-level audiences', '{"technical_depth": "low", "explanatory": true}', 82),
('audience_executive', 'Executive Audience Rule', 'audience', 'Tailors content for executive-level decision makers', '{"executive_summary": true, "strategic_focus": true}', 88),
('audience_technical', 'Technical Audience Rule', 'audience', 'Adapts for technical practitioners and specialists', '{"implementation_details": true, "technical_accuracy": true}', 86),
('audience_general', 'General Audience Rule', 'audience', 'Adapts for general public and broad audiences', '{"accessible_language": true, "broad_appeal": true}', 80),
('audience_academic', 'Academic Audience Rule', 'audience', 'Tailors content for academic and research audiences', '{"scholarly_tone": true, "research_focus": true}', 84),
('audience_student', 'Student Audience Rule', 'audience', 'Adapts content for student learning contexts', '{"learning_objectives": true, "pedagogical_approach": true}', 81),
('audience_customer', 'Customer Audience Rule', 'audience', 'Tailors content for customer-facing communications', '{"customer_focus": true, "value_proposition": true}', 83),
('audience_international', 'International Audience Rule', 'audience', 'Adapts for international and multicultural audiences', '{"cultural_sensitivity": true, "global_perspective": true}', 78),
('audience_stakeholder', 'Stakeholder Audience Rule', 'audience', 'Tailors content for various stakeholder groups', '{"stakeholder_analysis": true, "interest_alignment": true}', 87),

-- ===================================
-- Domain-Specific Rules
-- ===================================

('domain_technology', 'Technology Domain Rule', 'domain', 'Optimizes for technology and software contexts', '{"tech_terminology": true, "innovation_focus": true}', 85),
('domain_healthcare', 'Healthcare Domain Rule', 'domain', 'Adapts for healthcare and medical contexts', '{"medical_accuracy": true, "patient_safety": true}', 92),
('domain_finance', 'Finance Domain Rule', 'domain', 'Optimizes for financial and investment contexts', '{"financial_accuracy": true, "risk_awareness": true}', 89),
('domain_education', 'Education Domain Rule', 'domain', 'Adapts for educational and training contexts', '{"learning_outcomes": true, "pedagogical_soundness": true}', 84),
('domain_legal', 'Legal Domain Rule', 'domain', 'Optimizes for legal and regulatory contexts', '{"legal_precision": true, "compliance_focus": true}', 91),
('domain_marketing', 'Marketing Domain Rule', 'domain', 'Adapts for marketing and promotional contexts', '{"brand_alignment": true, "engagement_focus": true}', 82),
('domain_research', 'Research Domain Rule', 'domain', 'Optimizes for research and analysis contexts', '{"methodological_rigor": true, "evidence_based": true}', 86),
('domain_manufacturing', 'Manufacturing Domain Rule', 'domain', 'Adapts for manufacturing and production contexts', '{"operational_efficiency": true, "quality_focus": true}', 83),
('domain_retail', 'Retail Domain Rule', 'domain', 'Optimizes for retail and consumer contexts', '{"customer_experience": true, "sales_focus": true}', 81),
('domain_consulting', 'Consulting Domain Rule', 'domain', 'Adapts for consulting and advisory contexts', '{"solution_oriented": true, "client_value": true}', 87),

-- ===================================
-- Complexity Management Rules
-- ===================================

('complexity_simplify', 'Complexity Simplification Rule', 'complexity', 'Reduces complexity while maintaining effectiveness', '{"simplification_level": "moderate", "clarity_preservation": true}', 78),
('complexity_elaborate', 'Complexity Elaboration Rule', 'complexity', 'Adds necessary complexity and detail', '{"detail_expansion": true, "comprehensive_coverage": true}', 75),
('complexity_layer', 'Complexity Layering Rule', 'complexity', 'Organizes complex information in layers', '{"hierarchical_complexity": true, "progressive_disclosure": true}', 80),
('complexity_modular', 'Modular Complexity Rule', 'complexity', 'Breaks complex topics into modules', '{"modular_approach": true, "component_isolation": true}', 82),
('complexity_abstraction', 'Abstraction Management Rule', 'complexity', 'Manages abstraction levels appropriately', '{"abstraction_level": "optimal", "concrete_examples": true}', 79),
('complexity_synthesis', 'Complexity Synthesis Rule', 'complexity', 'Synthesizes complex information coherently', '{"information_integration": true, "coherent_narrative": true}', 81),
('complexity_prioritization', 'Complexity Prioritization Rule', 'complexity', 'Prioritizes complex elements by importance', '{"priority_ranking": true, "essential_focus": true}', 83),
('complexity_scaffolding', 'Complexity Scaffolding Rule', 'complexity', 'Provides scaffolding for complex topics', '{"learning_support": true, "gradual_complexity": true}', 77),
('complexity_visualization', 'Complexity Visualization Rule', 'complexity', 'Suggests visualization for complex concepts', '{"visual_aids": true, "diagram_recommendations": true}', 76),
('complexity_analogy', 'Complexity Analogy Rule', 'complexity', 'Uses analogies to explain complex concepts', '{"analogy_generation": true, "familiar_comparisons": true}', 74),

-- ===================================
-- Language and Grammar Rules
-- ===================================

('language_active_voice', 'Active Voice Rule', 'language', 'Promotes active voice over passive voice', '{"active_voice_preference": true, "clarity_improvement": true}', 70),
('language_conciseness', 'Conciseness Rule', 'language', 'Eliminates unnecessary words and phrases', '{"word_economy": true, "precision_focus": true}', 75),
('language_parallelism', 'Parallel Structure Rule', 'language', 'Ensures parallel structure in lists and series', '{"grammatical_consistency": true, "readability_improvement": true}', 68),
('language_transition', 'Transition Enhancement Rule', 'language', 'Improves transitions between ideas', '{"smooth_transitions": true, "logical_flow": true}', 72),
('language_vocabulary', 'Vocabulary Optimization Rule', 'language', 'Optimizes vocabulary for target audience', '{"vocabulary_level": "appropriate", "accessibility": true}', 73),
('language_sentence_variety', 'Sentence Variety Rule', 'language', 'Promotes varied sentence structures', '{"structural_variety": true, "rhythm_improvement": true}', 69),
('language_emphasis', 'Emphasis Enhancement Rule', 'language', 'Enhances emphasis and highlighting', '{"key_point_emphasis": true, "impact_maximization": true}', 74),
('language_precision', 'Language Precision Rule', 'language', 'Improves precision in word choice', '{"precise_terminology": true, "meaning_clarity": true}', 76),
('language_consistency', 'Language Consistency Rule', 'language', 'Maintains consistency in terminology', '{"term_consistency": true, "style_uniformity": true}', 71),
('language_inclusive', 'Inclusive Language Rule', 'language', 'Promotes inclusive and accessible language', '{"inclusive_terminology": true, "bias_reduction": true}', 77),

-- ===================================
-- Performance Optimization Rules
-- ===================================

('performance_response_time', 'Response Time Optimization Rule', 'performance', 'Optimizes for faster response generation', '{"speed_priority": "high", "efficiency_focus": true}', 85),
('performance_accuracy', 'Accuracy Optimization Rule', 'performance', 'Prioritizes accuracy over speed', '{"accuracy_priority": "high", "quality_focus": true}', 90),
('performance_balance', 'Balanced Performance Rule', 'performance', 'Balances speed and accuracy', '{"balance_mode": true, "optimal_tradeoff": true}', 87),
('performance_memory', 'Memory Efficiency Rule', 'performance', 'Optimizes for memory efficiency', '{"memory_optimization": true, "resource_awareness": true}', 82),
('performance_scalability', 'Scalability Optimization Rule', 'performance', 'Optimizes for scalable processing', '{"scalability_focus": true, "batch_processing": true}', 84),
('performance_caching', 'Caching Optimization Rule', 'performance', 'Optimizes for effective caching', '{"cache_friendly": true, "reusability": true}', 80),
('performance_parallel', 'Parallel Processing Rule', 'performance', 'Optimizes for parallel execution', '{"parallelization": true, "concurrent_processing": true}', 83),
('performance_bandwidth', 'Bandwidth Optimization Rule', 'performance', 'Optimizes for network efficiency', '{"bandwidth_awareness": true, "compression_friendly": true}', 78),
('performance_latency', 'Latency Minimization Rule', 'performance', 'Minimizes processing latency', '{"low_latency": true, "immediate_response": true}', 86),
('performance_throughput', 'Throughput Maximization Rule', 'performance', 'Maximizes processing throughput', '{"high_throughput": true, "batch_efficiency": true}', 81),

-- ===================================
-- Quality Assurance Rules
-- ===================================

('quality_validation', 'Quality Validation Rule', 'quality', 'Implements quality validation checks', '{"validation_level": "comprehensive", "quality_gates": true}', 88),
('quality_consistency', 'Quality Consistency Rule', 'quality', 'Ensures consistent quality standards', '{"consistency_checks": true, "standard_compliance": true}', 85),
('quality_completeness', 'Completeness Validation Rule', 'quality', 'Validates completeness of responses', '{"completeness_check": true, "requirement_coverage": true}', 87),
('quality_relevance', 'Relevance Validation Rule', 'quality', 'Validates relevance to user intent', '{"relevance_scoring": true, "intent_alignment": true}', 89),
('quality_accuracy', 'Accuracy Validation Rule', 'quality', 'Validates factual accuracy', '{"fact_checking": true, "accuracy_verification": true}', 92),
('quality_coherence', 'Coherence Validation Rule', 'quality', 'Validates logical coherence', '{"coherence_analysis": true, "logic_verification": true}', 86),
('quality_originality', 'Originality Validation Rule', 'quality', 'Validates originality and uniqueness', '{"originality_check": true, "plagiarism_detection": true}', 83),
('quality_appropriateness', 'Appropriateness Validation Rule', 'quality', 'Validates content appropriateness', '{"appropriateness_check": true, "content_filtering": true}', 84),
('quality_usability', 'Usability Validation Rule', 'quality', 'Validates practical usability', '{"usability_assessment": true, "practical_application": true}', 82),
('quality_feedback', 'Quality Feedback Integration Rule', 'quality', 'Integrates user feedback for quality improvement', '{"feedback_integration": true, "continuous_improvement": true}', 80),

-- ===================================
-- Innovation and Creativity Rules
-- ===================================

('innovation_creative', 'Creative Enhancement Rule', 'innovation', 'Enhances creative aspects while maintaining clarity', '{"creativity_boost": true, "innovation_focus": true}', 70),
('innovation_alternative', 'Alternative Perspective Rule', 'innovation', 'Introduces alternative perspectives and approaches', '{"perspective_diversity": true, "alternative_solutions": true}', 72),
('innovation_breakthrough', 'Breakthrough Thinking Rule', 'innovation', 'Promotes breakthrough and disruptive thinking', '{"disruptive_thinking": true, "paradigm_shift": true}', 68),
('innovation_synthesis', 'Innovation Synthesis Rule', 'innovation', 'Synthesizes ideas for innovative solutions', '{"idea_combination": true, "creative_synthesis": true}', 74),
('innovation_experimentation', 'Experimentation Rule', 'innovation', 'Encourages experimental approaches', '{"experimental_mindset": true, "hypothesis_testing": true}', 71),
('innovation_pattern_breaking', 'Pattern Breaking Rule', 'innovation', 'Breaks conventional patterns and assumptions', '{"assumption_challenging": true, "pattern_disruption": true}', 69),
('innovation_cross_pollination', 'Cross-Pollination Rule', 'innovation', 'Applies insights from other domains', '{"cross_domain_insights": true, "knowledge_transfer": true}', 73),
('innovation_future_thinking', 'Future Thinking Rule', 'innovation', 'Incorporates future-oriented perspectives', '{"future_orientation": true, "trend_awareness": true}', 75),
('innovation_constraint_removal', 'Constraint Removal Rule', 'innovation', 'Identifies and removes limiting constraints', '{"constraint_analysis": true, "limitation_removal": true}', 67),
('innovation_serendipity', 'Serendipity Enhancement Rule', 'innovation', 'Creates opportunities for serendipitous discoveries', '{"serendipity_promotion": true, "unexpected_connections": true}', 66),

-- ===================================
-- Security and Privacy Rules
-- ===================================

('security_privacy', 'Privacy Protection Rule', 'security', 'Ensures privacy protection in prompts', '{"privacy_preservation": true, "data_protection": true}', 95),
('security_confidentiality', 'Confidentiality Rule', 'security', 'Maintains confidentiality requirements', '{"confidentiality_level": "high", "information_classification": true}', 93),
('security_compliance', 'Security Compliance Rule', 'security', 'Ensures compliance with security standards', '{"compliance_verification": true, "standard_adherence": true}', 91),
('security_access_control', 'Access Control Rule', 'security', 'Implements appropriate access control considerations', '{"access_levels": true, "authorization_awareness": true}', 89),
('security_audit_trail', 'Audit Trail Rule', 'security', 'Maintains audit trail considerations', '{"audit_logging": true, "traceability": true}', 87),
('security_encryption', 'Encryption Awareness Rule', 'security', 'Considers encryption requirements', '{"encryption_awareness": true, "data_security": true}', 88),
('security_vulnerability', 'Vulnerability Assessment Rule', 'security', 'Identifies potential security vulnerabilities', '{"vulnerability_scanning": true, "risk_assessment": true}', 90),
('security_incident', 'Incident Response Rule', 'security', 'Incorporates incident response considerations', '{"incident_preparedness": true, "response_planning": true}', 86),
('security_authentication', 'Authentication Rule', 'security', 'Ensures proper authentication considerations', '{"authentication_methods": true, "identity_verification": true}', 92),
('security_monitoring', 'Security Monitoring Rule', 'security', 'Implements security monitoring requirements', '{"monitoring_capabilities": true, "threat_detection": true}', 88),

-- ===================================
-- Accessibility and Inclusion Rules
-- ===================================

('accessibility_universal', 'Universal Accessibility Rule', 'accessibility', 'Ensures universal accessibility principles', '{"universal_design": true, "barrier_removal": true}', 85),
('accessibility_visual', 'Visual Accessibility Rule', 'accessibility', 'Addresses visual accessibility needs', '{"visual_impairment": true, "screen_reader_friendly": true}', 83),
('accessibility_auditory', 'Auditory Accessibility Rule', 'accessibility', 'Addresses auditory accessibility needs', '{"hearing_impairment": true, "alternative_formats": true}', 82),
('accessibility_cognitive', 'Cognitive Accessibility Rule', 'accessibility', 'Addresses cognitive accessibility needs', '{"cognitive_load": true, "clear_navigation": true}', 84),
('accessibility_motor', 'Motor Accessibility Rule', 'accessibility', 'Addresses motor accessibility needs', '{"motor_impairment": true, "alternative_input": true}', 81),
('accessibility_language', 'Language Accessibility Rule', 'accessibility', 'Ensures language accessibility', '{"plain_language": true, "translation_friendly": true}', 86),
('accessibility_cultural', 'Cultural Accessibility Rule', 'accessibility', 'Ensures cultural accessibility and sensitivity', '{"cultural_inclusivity": true, "bias_awareness": true}', 87),
('accessibility_economic', 'Economic Accessibility Rule', 'accessibility', 'Considers economic accessibility factors', '{"cost_awareness": true, "resource_efficiency": true}', 80),
('accessibility_technological', 'Technological Accessibility Rule', 'accessibility', 'Ensures technological accessibility', '{"device_compatibility": true, "bandwidth_awareness": true}', 79),
('accessibility_documentation', 'Accessibility Documentation Rule', 'accessibility', 'Ensures accessible documentation practices', '{"accessible_formats": true, "clear_instructions": true}', 88),

-- ===================================
-- Additional Specialized Rules (to meet 185+ requirement)
-- ===================================

-- Communication Enhancement Rules
('communication_brevity', 'Communication Brevity Rule', 'communication', 'Promotes concise communication', '{"word_economy": true, "essential_only": true}', 75),
('communication_empathy', 'Empathy Communication Rule', 'communication', 'Adds empathetic communication elements', '{"emotional_awareness": true, "understanding": true}', 78),
('communication_urgency', 'Urgency Communication Rule', 'communication', 'Adjusts urgency level appropriately', '{"urgency_indicators": true, "priority_clarity": true}', 80),
('communication_feedback', 'Feedback Communication Rule', 'communication', 'Optimizes feedback delivery', '{"constructive_approach": true, "actionable_suggestions": true}', 82),
('communication_conflict', 'Conflict Resolution Communication Rule', 'communication', 'Addresses conflict resolution needs', '{"neutral_tone": true, "solution_focus": true}', 84),

-- Workflow Enhancement Rules
('workflow_automation', 'Workflow Automation Rule', 'workflow', 'Identifies automation opportunities', '{"automation_potential": true, "efficiency_gains": true}', 77),
('workflow_standardization', 'Workflow Standardization Rule', 'workflow', 'Promotes workflow standardization', '{"process_consistency": true, "best_practices": true}', 79),
('workflow_optimization', 'Workflow Optimization Rule', 'workflow', 'Optimizes workflow efficiency', '{"bottleneck_identification": true, "process_improvement": true}', 81),
('workflow_collaboration', 'Collaboration Workflow Rule', 'workflow', 'Enhances collaborative workflows', '{"team_coordination": true, "shared_resources": true}', 76),
('workflow_documentation', 'Workflow Documentation Rule', 'workflow', 'Ensures proper workflow documentation', '{"process_documentation": true, "knowledge_capture": true}', 78),

-- Decision Support Rules
('decision_criteria', 'Decision Criteria Rule', 'decision', 'Clarifies decision criteria', '{"criteria_definition": true, "evaluation_framework": true}', 85),
('decision_alternatives', 'Decision Alternatives Rule', 'decision', 'Identifies decision alternatives', '{"option_generation": true, "alternative_analysis": true}', 83),
('decision_risk', 'Decision Risk Assessment Rule', 'decision', 'Assesses decision risks', '{"risk_identification": true, "mitigation_strategies": true}', 87),
('decision_stakeholder', 'Decision Stakeholder Rule', 'decision', 'Considers stakeholder impact', '{"stakeholder_analysis": true, "impact_assessment": true}', 86),
('decision_timeline', 'Decision Timeline Rule', 'decision', 'Establishes decision timelines', '{"timeline_clarity": true, "deadline_awareness": true}', 82),

-- Learning Enhancement Rules
('learning_objectives', 'Learning Objectives Rule', 'learning', 'Defines clear learning objectives', '{"objective_clarity": true, "measurable_outcomes": true}', 84),
('learning_progression', 'Learning Progression Rule', 'learning', 'Ensures logical learning progression', '{"skill_building": true, "prerequisite_awareness": true}', 82),
('learning_assessment', 'Learning Assessment Rule', 'learning', 'Incorporates assessment methods', '{"assessment_design": true, "progress_tracking": true}', 85),
('learning_feedback', 'Learning Feedback Rule', 'learning', 'Provides effective learning feedback', '{"feedback_quality": true, "improvement_guidance": true}', 83),
('learning_engagement', 'Learning Engagement Rule', 'learning', 'Enhances learner engagement', '{"engagement_strategies": true, "motivation_building": true}', 81),

-- Project Management Rules
('project_scope', 'Project Scope Rule', 'project', 'Clarifies project scope boundaries', '{"scope_definition": true, "boundary_clarity": true}', 88),
('project_timeline', 'Project Timeline Rule', 'project', 'Establishes realistic project timelines', '{"timeline_planning": true, "milestone_definition": true}', 86),
('project_resources', 'Project Resources Rule', 'project', 'Identifies required project resources', '{"resource_planning": true, "capacity_analysis": true}', 84),
('project_risks', 'Project Risk Management Rule', 'project', 'Identifies and manages project risks', '{"risk_assessment": true, "contingency_planning": true}', 87),
('project_communication', 'Project Communication Rule', 'project', 'Establishes project communication protocols', '{"communication_plan": true, "stakeholder_updates": true}', 85),

-- Data Management Rules
('data_quality', 'Data Quality Rule', 'data', 'Ensures data quality standards', '{"quality_criteria": true, "validation_methods": true}', 90),
('data_governance', 'Data Governance Rule', 'data', 'Implements data governance practices', '{"governance_framework": true, "compliance_tracking": true}', 89),
('data_privacy', 'Data Privacy Rule', 'data', 'Ensures data privacy protection', '{"privacy_compliance": true, "data_protection": true}', 92),
('data_integration', 'Data Integration Rule', 'data', 'Manages data integration requirements', '{"integration_standards": true, "data_mapping": true}', 86),
('data_lifecycle', 'Data Lifecycle Rule', 'data', 'Manages data lifecycle processes', '{"lifecycle_management": true, "retention_policies": true}', 88),

-- User Experience Rules
('ux_usability', 'UX Usability Rule', 'ux', 'Ensures usability principles', '{"usability_testing": true, "user_centered": true}', 85),
('ux_accessibility', 'UX Accessibility Rule', 'ux', 'Incorporates accessibility in UX', '{"accessibility_standards": true, "inclusive_design": true}', 87),
('ux_responsiveness', 'UX Responsiveness Rule', 'ux', 'Ensures responsive design', '{"responsive_design": true, "device_compatibility": true}', 83),
('ux_performance', 'UX Performance Rule', 'ux', 'Optimizes user experience performance', '{"performance_optimization": true, "load_time_awareness": true}', 86),
('ux_feedback', 'UX Feedback Rule', 'ux', 'Incorporates user feedback mechanisms', '{"feedback_collection": true, "iterative_improvement": true}', 84),

-- Content Strategy Rules  
('content_seo', 'Content SEO Rule', 'content', 'Optimizes content for search engines', '{"seo_optimization": true, "keyword_integration": true}', 79),
('content_readability', 'Content Readability Rule', 'content', 'Ensures content readability', '{"readability_score": true, "comprehension_level": true}', 82),
('content_engagement', 'Content Engagement Rule', 'content', 'Maximizes content engagement', '{"engagement_techniques": true, "audience_connection": true}', 80),
('content_consistency', 'Content Consistency Rule', 'content', 'Maintains content consistency', '{"brand_voice": true, "style_guide": true}', 81),
('content_freshness', 'Content Freshness Rule', 'content', 'Ensures content freshness and relevance', '{"update_frequency": true, "relevance_monitoring": true}', 78),

-- Final Rules to Meet 185+ Requirement
('integration_api', 'API Integration Rule', 'integration', 'Optimizes API integration approaches', '{"api_design": true, "integration_patterns": true}', 84),
('integration_database', 'Database Integration Rule', 'integration', 'Manages database integration requirements', '{"data_consistency": true, "transaction_management": true}', 86),
('integration_security', 'Security Integration Rule', 'integration', 'Ensures secure integration practices', '{"security_protocols": true, "authentication_integration": true}', 89),
('integration_monitoring', 'Monitoring Integration Rule', 'integration', 'Integrates monitoring and observability', '{"monitoring_setup": true, "alerting_integration": true}', 85),
('integration_testing', 'Testing Integration Rule', 'integration', 'Ensures comprehensive integration testing', '{"test_automation": true, "integration_coverage": true}', 87);

-- ===================================
-- Rule Performance Tracking Setup
-- ===================================

-- Insert sample rule performance data for testing
INSERT INTO rule_performance (rule_id, rule_name, prompt_type, improvement_score, confidence_level, execution_time_ms, rule_parameters) VALUES
('clarity_basic', 'Basic Clarity Enhancement', 'technical', 0.85, 0.92, 150, '{"applied": true, "context": "database_query"}'),
('specificity_basic', 'Basic Specificity Enhancement', 'business', 0.78, 0.88, 120, '{"applied": true, "context": "requirements"}'),
('structure_basic', 'Basic Structure Enhancement', 'educational', 0.82, 0.90, 180, '{"applied": true, "context": "tutorial"}'),
('context_basic', 'Basic Context Enhancement', 'creative', 0.73, 0.85, 200, '{"applied": true, "context": "storytelling"}');

-- ===================================
-- Completion and Statistics
-- ===================================

-- Log the completion of rule seeding
DO $$
DECLARE
    rule_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO rule_count FROM rule_metadata;
    
    RAISE NOTICE 'Rule metadata seeding completed successfully!';
    RAISE NOTICE 'Total rules seeded: %', rule_count;
    RAISE NOTICE 'Seeding timestamp: %', NOW();
    
    -- Verify we have the required 185+ rules
    IF rule_count >= 185 THEN
        RAISE NOTICE 'SUCCESS: Rule seeding meets requirement of 185+ rules';
    ELSE
        RAISE WARNING 'WARNING: Only % rules seeded, requirement is 185+', rule_count;
    END IF;
END $$;