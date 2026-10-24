"""Prompts extracted from the accepted inference code; no model or file I/O."""
from __future__ import annotations


def _get_dynamic_prompt(frame_number: int) -> tuple[str, str]:
    """Get system and task instructions for DYNAMIC descriptions (with arrows)."""
    system_instruction = f'''
        You are a pattern recognition expert. You will analyze multiple frame descriptions at the same time position to find their common pattern.
    '''
    
    # task_instruction = f'''
    #     INPUT:
    #     You will receive multiple descriptions of Frame {frame_number} from different video sets.
    #     All these frames show the SAME physical action but from different instances.

    #     CRITICAL RULES:
    #     1. Only observe and summarize/record what is present. Do NOT predict or infer. Only use what is explicitly described.
    #     2. Use the original wording from the descriptions as much as possible. Do not paraphrase or invent new phrases.
        
    #     HANDLING SETTINGS VARIATIONS:
    #     - Variations across different videos (e.g., different object colors, sizes, backgrounds, text on objects) are normal settings differences.
    #     - If text/labels on objects are consistent across all instances, treat them as settings variation (not important to pattern).
    #     - However, if text/labels CHANGE across frames within the SAME video, this IS part of the pattern and must be captured.
    #     - Include all setting variants using actual words from descriptions (e.g., "red ball / blue ball / green ball").
        
    #     ARROWS - HIGHEST PRIORITY (describe in complete detail):
    #     - Arrows are always ORANGE color.
    #     - For EACH arrow, you must describe ALL of these:
    #       * Which PART of the object the arrow is associated with (e.g. "arrow on the top part of the bottle", "arrow near the base of the box")
    #       * Direction arrow points (up/down/left/right/diagonal, etc.)
    #       * Relative position to the object (above/below/beside/on/near the object part)
    #       * Size if mentioned (large/small - larger arrows indicate stronger/faster movement)
    #     - If there are multiple arrows on DIFFERENT PARTS of the object, describe EACH one separately with its associated part.
    #     - Example: "Orange arrow on the top part of the bottle pointing downward, orange arrow on the bottom near the base pointing left"
        
    #     ACTIONS - describe all motions and state changes:
    #     - Capture all actions mentioned (pushing, pulling, rotating, tilting, opening, closing, etc.)
    #     - Include object states (moving/still, open/closed, compressed/extended, etc.)
    #     - Use the exact action words from the descriptions.
        
    #     YOUR TASK:
    #     Write a comprehensive pattern that includes:
    #     - All objects with their setting variations
    #     - All actions and motions (using original words)
    #     - Complete arrow details for each arrow, including which part of the object it's on
    #     - Spatial relationships between objects
        
    #     Output exactly one line in this format:
    #     <Frame {frame_number}>: [comprehensive pattern using original text, including all objects, actions, complete arrow details with object parts, and setting variations]
    # '''
    

    task_instruction = f'''
        INPUT:
        You will receive multiple descriptions of Frame {frame_number} from different video sets.
        All these frames show the SAME physical action but from different instances.

        CRITICAL RULES:
        1. **Summarize the general pattern** based on what is explicitly described. Do NOT predict or infer.
        2. **Prioritize brevity and clarity.** Capture the main objects and actions.
        3. Use concise, descriptive language. You may synthesize original wording to create a clear, general pattern. **Do not include source references (e.g., "D1", "D5").**
        
        HANDLING VARIATIONS:
        - **Generalize non-arrow setting variations.** For settings like object color, size, or background, use a general term (e.g., "a ball of various colors") instead of listing all variants.
        - **Arrow variations are critical** and must be captured in detail, as described in the section below.
        
        ARROWS (HIGH PRIORITY):
        - **All arrow variations are critical and MUST be captured.** This is the most important part of the pattern.
        - **If arrow states differ significantly (e.g., some instances have "no arrows," while others have "a few arrows," and others have "numerous arrows"), all of these distinct states must be described** as part of the general pattern.
        - Arrows are always ORANGE.
        - For instances *with* arrows, **summarize the arrows' general placement and direction.**
        - Instead of listing every single arrow, describe the **overall pattern** for each state (e.g., "one state has no arrows; another state has numerous short arrows...").
        - **Mention the key object parts** the arrows are associated with (e.g., "arrows on the bumper and wheels").
        - **Describe the general directions** (e.g., "pointing mostly outward").
        - Include size (short/medium/long) if it's a consistent part of the pattern.
        - Example: "Pattern includes three states: one with no arrows; one with a few isolated arrows at the corners; and one with numerous short arrows scattered across the body, pointing outward."
        
        ACTIONS:
        - Capture all actions mentioned (pushing, pulling, rotating, etc.).
        - Include key object states (moving/still, open/closed, etc.).
        
        YOUR TASK:
        Write a **concise, general pattern** that includes:
        - The main objects.
        - All actions and motions.
        - A summarized description of all arrow patterns/states (placement, direction, size, quantity).
        
        Output exactly one line in this format:
        <Frame {frame_number}>: [concise general pattern capturing main objects, actions, summarized arrow patterns, and important variations]'''
    
    return system_instruction, task_instruction

def _get_optical_flow_prompt() -> tuple[str, str]:
    """Get system and task instructions for OPTICAL FLOW descriptions."""
    system_instruction = f'''
        You are a pattern recognition expert specializing in optical flow analysis.
        You will analyze multiple optical flow descriptions for the same type of video to find their common motion pattern in rules.
    '''
    
    task_instruction = f'''
        INPUT:
        You will receive multiple data summaries for the same type of video.
        All summaries follow the same 13-field format.

        CRITICAL RULES:
        1. **Summarize the general pattern** across all provided data summaries.
        2. **Your output MUST follow the *exact* 13-field format** of the input.
        3. Use concise, descriptive language. You may **synthesize original wording** to create a clear, general pattern.
        4. **Do not include source references** (e.g., "Data 1," "Data 2").
        
        DATA FIELD GENERALIZATION (HIGH PRIORITY):
        - **This is the most important part of the pattern.** You must generate a generalized entry for *each* of the 13 fields.
        - **Numerical Fields (`max`, `min`, `peak number`, `valley number`):**
          - If the value is consistent across all inputs (e.g., all say "peak number: 3"), use that specific value (e.g., "peak number: 3").
          - If the values vary, summarize them as a **range** (e.g., "max: 0.8-1.0", "min: 0.1-0.2", "peak number: 2-4").
        - **Text-Based Fields (all other fields):**
          - If the description is consistent across all inputs (e.g., all say "trend: No clear trend"), use that exact description.
          - **Variations are critical and MUST be captured.** If descriptions vary (e.g., some say "vibrating: Yes..." while others say "vibrating: No..."), you **must capture all variations** using a slash ( / ) (e.g., "vibrating: Yes, periodic up-down motion / No, smooth motion").
          - Example: "stability: Fluctuating / Stable" or "overall smoothness: Moderate / Smooth".
              - Condense the `summary` fields into a single, representative summary of the *overall* pattern.
        
        YOUR TASK:
        Write a **single, comprehensive pattern** that:
        1. Exactly follows the 13-field format.
        2. Provides the generalized range for all numerical fields.
        3. Provides the generalized description (including all variations separated by "/") for all text fields.
        
        Output the 13-field pattern directly, starting with "max:":
        max: [generalized value or range]
        min: [generalized value or range]
        peak number: [generalized value or range]
        valley number: [generalized value or range]
        peak description: [generalized description, capturing all variations]
        near-still motions: [generalized description, capturing all variations]
        trend (increase/decrease/any other trend): [generalized description, capturing all variations]
        vibrating (frequent up-down): [generalized description, capturing all variations]
        overall smoothness: [generalized description, capturing all variations]
        stability: [generalized description, capturing all variations]
        notable transitions: [generalized description, capturing all variations]
        value change summary: [generalized summary description]
        summary: [generalized summary description]
      '''
    
    return system_instruction, task_instruction

def _get_dynamic_anomaly_prompt() -> tuple[str, str]:
    """Get prompts for dynamic anomaly detection (with arrows)."""
    system_instruction = '''
        You are an anomaly detection expert. Your job is to compare a frame description set against a pattern and detect if there are any anomalies.
    '''
    
    task_instruction = '''
        INPUT:
        - A frame description set with 8 frames (Frame 1 to Frame 8), in time order
        - A pattern showing what NORMALLY happens in this physical action
        
        WHAT IS ANOMALY:
        Anomaly means something STRANGE or UNEXPECTED happened in the video compared to the pattern.
        This includes:
        - Unexpected objects appearing
        - Unexpected or different motions/actions
        - Strange movements that don't match the pattern
        
        ARROWS ARE CRITICAL - USE THEM TO UNDERSTAND MOVEMENT:
        - All arrows are ORANGE color
        - Arrows show the FLOW and DIRECTION of movement - use them to understand how objects should move
        - Check arrow direction to understand what part of the object can move and how
        - If arrows are MISSING or EXTRA compared to the pattern → ANOMALY
        - If arrows point differently than pattern → wrong movement → ANOMALY
        - If arrows are on different parts of the object than pattern → ANOMALY
        - If arrow size differs significantly (large vs small) → different movement magnitude → potential ANOMALY
        
        YOUR TASK:
        1. Look at the pattern's arrows to understand the expected movement flow
        2. Look at the frame's arrows to see the actual movement flow
        3. Compare: Do arrows match (direction, position on object, size)?
        4. Compare: Are objects and actions the same?
        5. If anything is different or strange → ANOMALY
        
        SCORING:
        - 0.0-0.2 = No anomaly (everything matches)
        - 0.3-0.5 = Minor anomaly (small differences)
        - 0.6-0.8 = Moderate anomaly (clear differences in movement or objects)
        - 0.9-1.0 = Severe anomaly (completely wrong movement/action or wrong arrow direction/position)
        
        OUTPUT FORMAT:
        Brief reasoning, then end with:
        Anomaly Score: <your score>
        
        Do not write anything after the score line.
    '''
    
    return system_instruction, task_instruction

def _get_optical_flow_anomaly_prompt() -> tuple[str, str]:
    """Get prompts for optical flow anomaly detection."""
    system_instruction = '''
        You are an anomaly detection expert for optical flow analysis. Your job is to compare an optical flow data summary against a pattern and detect if there are any anomalies.
    '''
    
    task_instruction = '''
        INPUT:
        - An optical flow data summary (13 fields) for a single video
        - A pattern showing what NORMALLY happens in this type of video (also 13 fields)
        
        WHAT IS ANOMALY IN OPTICAL FLOW:
        Anomaly means the motion pattern in the video is STRANGE or UNEXPECTED compared to the normal pattern.
        This includes:
        - UNEXPECTED MOTION CHARACTERISTICS: Different max/min values, different peak/valley counts
        - ABNORMAL TRENDS: Different increase/decrease patterns than expected
        - UNUSUAL VIBRATIONS: Vibrating when shouldn't, or smooth when should vibrate
        - WRONG STABILITY: Unstable when should be stable, or vice versa
        - DIFFERENT SMOOTHNESS: Jerky when should be smooth, or vice versa
        - ANOMALOUS TRANSITIONS: Unexpected sudden changes or missing expected transitions
        
        HOW TO COMPARE OPTICAL FLOW DATA:
        
        1. NUMERICAL FIELDS (max, min, peak number, valley number):
           - Check if values fall within the expected range from the pattern
           - Small deviations (±10-20%) may be normal variation
           - Large deviations (>50%) indicate anomaly
        
        2. TEXT FIELDS (descriptions of motion characteristics):
           - Compare each field description to the pattern
           - If pattern shows variations (e.g., "Yes / No"), the video should match ONE of them
           - If video shows something NOT in the pattern variations → ANOMALY
        
        3. OVERALL MOTION BEHAVIOR:
           - Does the summary description match the pattern's expected behavior?
           - Are the motion characteristics (smooth/jerky, stable/unstable, etc.) consistent?
        
        YOUR TASK:
        Compare the optical flow data to the pattern field by field:
        1. Check numerical values against expected ranges
        2. Check text descriptions against expected variations
        3. Assess overall motion behavior consistency
        4. If significant differences exist → ANOMALY
        
        SCORING GUIDE:
        - 0.0-0.2 = No anomaly (all fields match pattern or within normal variation)
        - 0.3-0.5 = Minor anomaly (1-2 fields show small deviations)
        - 0.6-0.8 = Moderate anomaly (several fields differ, or important fields show clear deviations)
        - 0.9-1.0 = Severe anomaly (many fields wrong, or critical characteristics completely different)
        
        OUTPUT FORMAT:
        Brief reasoning comparing each field, then end with:
        Anomaly Score: <your score>
        
        Do not write anything after the score line.
    '''
    
    return system_instruction, task_instruction

def build_system_prompt() -> str:
    return (
        "You are an Industrial Video Anomaly Scoring Expert.\n"
        "Your PRIMARY goal is to assign a reliable anomaly score to a test clip by "
        "synthesizing multi-modal textual reasonings that compare the clip against normal training patterns.\n"
        "Carefully consider: motion strength, direction, periodicity/rhythm, object state changes, stability, timing, "
        "and appearance consistency.\n"
        "\n"
        "OUTPUT REQUIREMENTS (IMPORTANT):\n"
        "1) First, output the ORIGINAL technical reasoning and the anomaly score:\n"
        "   - 'Reasoning: ' followed by a concise 2–3 sentence explanation based on the modality reasonings.\n"
        "   - 'Anomaly Score: ' followed by ONE number in [0.0,1.0].\n"
        "   This score is the ONLY anomaly score you output and is used for evaluation.\n"
        "2) After that, output ONE extra field:\n"
        "   - 'Final Explanation: ' followed by 2–4 sentences of human-friendly description that explains, in plain language, "
        "     what happens to the object in the video and why this behavior is normal or abnormal.\n"
        "\n"
        "STRICT FORMAT:\n"
        "- You MUST output exactly one line starting with 'Anomaly Score:'. Do NOT output a second anomaly score.\n"
        "- Do NOT change the anomaly score when writing the Final Explanation; it must be consistent with your original reasoning.\n"
        "- Do NOT output anything after the 'Final Explanation:' text block.\n"
    )

MODALITY_NOTES = {
    "dynamic": "- **Dynamic Source Analysis**: motion-annotated frame analysis (arrows showing how objects move).",
    "oflow":   "- **Optical Flow Source Analysis**: motion magnitude / optical-flow based descriptions.",
    "static":  "- **Static Source Analysis**: raw frame appearance descriptions."
}

def build_user_prompt(
    mod_sections: list[tuple[str, str]],
    modality_list: list[str],
    object_name: str | None = None,
    description_text: str | None = None
) -> str:
    notes_lines = [MODALITY_NOTES[m] for m in modality_list if m in MODALITY_NOTES]
    notes = ("Modalities provided:\n" + "\n".join(notes_lines) + "\n\n") if notes_lines else ""

    # Object naming guidance
    if object_name:
        obj_naming = (
            f"Use the term \"{object_name}\" when referring to the main object in the clip.\n"
        )
    else:
        obj_naming = (
            "If the main object type is mentioned in the inputs (e.g., ball, USB drive, button, gear), "
            "use that exact name consistently. Otherwise, choose a simple, reasonable name such as "
            "\"the object\" or \"the device\".\n"
        )

    header = (
        "INPUT:"
        "You will receive:"
        "1. **Raw Frame Description**: The visual description of the original video frames (8 frames)."
        "   - This is what we're trying to explain - the actual visual content of the video."
        
        "2. **Dynamic Source Analysis**:"
        "   - Anomaly reasoning: Explanation from dynamic description analysis (includes arrow motion indicators)"
        
        "3. **Optical Flow Source Analysis**:"
        "   - Anomaly reasoning: Explanation from optical flow motion pattern analysis"

        "YOUR TASK:\n"
        "- First, read the **Raw Frame Description** carefully to understand the visual and physical setting of the video.\n"
        "- Then, integrate the evidence from ALL other modalities to infer whether the clip behaves normally or abnormally.\n"
        "- Treat strong consistency with expected physical or behavioral patterns as evidence of normality.\n"
        "- Treat deviations in motion, stability, rhythm, or object response as evidence of anomaly.\n"
        "- Use all information holistically; do not rely solely on any single modality.\n\n"

        "REASONING REQUIREMENT (FOR THE FIRST 'Reasoning:' FIELD):\n"
        "- Write a concise 2–3 sentence summary combining the Raw Frame Description and Dynamic Source Analysis and Optical Flow Source Analysis reasonings.\n"
        "- Focus on the *physical or behavioral correctness* of the object’s motion, timing, and stability.\n"
        "- Clearly conclude whether the clip overall appears **normal** or **anomalous**, and briefly justify why.\n\n"

        "SCORING GUIDE (FOR 'Anomaly Score:'):\n"
        "- 0.0–0.2: No anomaly (fully consistent with normal behavior).\n"
        "- 0.3–0.5: Minor anomaly (1–2 mild irregularities; generally functional).\n"
        "- 0.6–0.8: Moderate anomaly (clear abnormal pattern or multiple deviations).\n"
        "- 0.9–1.0: Severe anomaly (major malfunction or very different motion pattern).\n\n"

        "FINAL EXPLANATION REQUIREMENT (HUMAN-FRIENDLY):\n"
        f"- {obj_naming}"
        "- In 'Final Explanation:', write a short, clear summary in everyday language.\n"
        "- The first sentence must explicitly state whether the clip is normal, roughly normal, slightly anomalous, or anomalous "
        "(e.g., 'This is normal.', 'This is roughly normal.', 'This is slightly anomalous.', or 'This is anomalous.').\n"
        "- Follow with 1–2 sentences (no more than 3 total) explaining why, based on what happens in the video.\n"
        "- Combine understanding from the Static Source and all modality analyses.\n"
        "- Clearly describe what happens to the object (e.g., moves smoothly, vibrates irregularly, gets stuck, deforms too much, fails to return, etc.).\n"
        "- Do **not** mention specific modalities, arrows, plots, metrics, or scores.\n"
        "- Keep the explanation intuitive, concise, and consistent with the anomaly score and the overall reasoning.\n\n"

        "OUTPUT FORMAT (STRICT):\n"
        "Reasoning: <2–3 sentence technical summary combining raw frame + multi-modal reasoning>\n"
        "Anomaly Score: <one number in [0.0,1.0]>\n"
        "Final Explanation: <2–4 sentences in natural language describing the object’s behavior and why it is normal or abnormal>\n"
        "Do not output any additional sections or commentary.\n"
        "Only output exactly three lines: 'Reasoning:', 'Anomaly Score:', and 'Final Explanation:'.\n"
    )

    body_parts = []
    for title, text in mod_sections:
        body_parts.append(f"{title}:\n{text}\n")
    
    final_prompt = header + "\n**Raw Frame Description**\n" + description_text + "\n" + "\n".join(body_parts)
    return final_prompt

def static_caption_prompt(obj):
    # video2text.py, not use_arrow_video branch
    return f'Describe this image in detail, focusing on main objects, actions, and the surroundings of the main objects. The image may contains anomaly, so you should only describe the image from what you see, do not make assumptions of physics. The main object is {obj}.'

def dynamic_caption_prompt(obj):
    # video2text.py, use_arrow_video branch
    return f'''
                        The main object is {obj}. Analyze the image and describe any orange arrows or indicators. For each arrow, include:
                        - Location: Where on the object the arrow appears (e.g., central pin, mounting holes, edges, left/right plates).
                        - Direction: What the arrow points to or highlights (inward, outward, up, down, specific feature, etc.).
                        - Length: Short, medium, or long, relative to the object's features.
                        - Quantity & Distribution: How many arrows are present and how they are arranged (few/several/numerous, scattered, clustered, radiating, etc.).
                        If there are no arrows or orange indicators, clearly state: "There are no orange arrows or indicators of movement in the image."
                        Be specific and thorough in your description, treating arrows as overlays highlighting important features.
                        '''


def _get_static_prompt(frame_number: int) -> tuple[str, str]:
    """Get system and task instructions for STATIC descriptions (no arrows)."""
    system_instruction = f'''
        You are a pattern recognition expert specializing in visual analysis. You will analyze multiple static frame descriptions at the same time position to find their common visual pattern.
    '''
    
    task_instruction = f'''
        INPUT:
        You will receive multiple STATIC descriptions of Frame {frame_number} from different video sets.
        These are visual snapshots WITHOUT any motion arrows or directional indicators.
        All frames show the SAME physical scene/object but from different instances.

        CRITICAL RULES:
        1. Only observe and summarize what is VISUALLY PRESENT. Do NOT predict, infer actions, or imagine motion.
        2. Use the original wording from the descriptions as much as possible. Do not paraphrase.
        3. NO ARROWS exist in static descriptions - do not mention or look for arrows.
        
        FOCUS ON VISUAL DETAILS:
        The descriptions will likely identify a MAIN OBJECT. For this main object, elaborate on:
        - Physical appearance (shape, size, color, material, texture)
        - Current state/configuration (open/closed, extended/compressed, position, orientation)
        - Parts and components visible (base, top, handle, lid, buttons, etc.)
        - Spatial position (where it is in the frame, what it's on/in)
        - Surface details (markings, text, labels, patterns)
        - Any other objects present and their relationship to the main object
        
        HANDLING VARIATIONS:
        - Different instances may have different colors, sizes, or text labels - these are setting variations.
        - Include all visual variants using original words (e.g., "red bottle / blue bottle / green bottle").
        - If text/labels are consistent across instances, treat as setting detail (include the variation).
        - Focus on WHAT IS VISIBLE, not what is happening or will happen.
        
        DESCRIBE THE VISUAL SCENE:
        - What objects are present and visible
        - How they look (appearance details)
        - Where they are positioned
        - Their current physical state/configuration
        - Relationships between objects (spatial arrangement)
        
        YOUR TASK:
        Write a comprehensive VISUAL pattern that captures:
        - The main object with ALL its visual details
        - All visible parts and their appearance
        - Current states and configurations
        - Setting variations (colors, sizes, labels, etc.)
        - Other objects and their spatial relationships
        
        Use original wording. Be thorough about visual details.
        
        Output exactly one line in this format:
        <Frame {frame_number}>: [detailed visual pattern describing what is seen, using original text, with all object details and setting variations]
    '''

    task_instruction = f'''
        INPUT:
        You will receive multiple STATIC descriptions of Frame {frame_number} from different video sets.
        These are visual snapshots WITHOUT any motion arrows or directional indicators.
        All frames show the SAME physical scene/object but from different instances.

        CRITICAL RULES:
        1. **Summarize the general pattern** based on what is VISUALLY PRESENT. Do NOT predict, infer actions, or imagine motion.
        2. **Prioritize brevity and clarity.** Capture the main objects and their static state.
        3. Use concise, descriptive language. You may **synthesize original wording** to create a clear, general pattern. **Do not include source references (e.g., "D1", "D5").**
        4. **NO ARROWS exist in static descriptions.** Your focus is entirely on the objects themselves.
        
        STATIC DETAILS (HIGH PRIORITY):
        - **This is the most important part of the pattern.**
        - **Focus on the object's configuration and key visual features.**
        - Instead of listing every minor detail, summarize the **defining visual characteristics** and **static state.**
        - **Mention key object parts** and their state (e.g., "a box with its lid open," "a switch in the 'on' position," "a compressed spring").
        - **Describe the general position** (e.g., "on a surface," "held by a hand").
        - Example: "A box of various colors, with its lid open, resting on a table. Text is visible on the front panel."
        
        HANDLING VARIATIONS:
        - **Generalize common setting variations.** For non-structural settings (e.g., object color, size, background, minor text), use a general term (e.g., "a box of various colors").
        - **Structural or State variations are critical and MUST be captured.** If descriptions show fundamentally different *static states* or *configurations* (e.g., some instances show the "lid open" while others show the "lid closed"), all of these distinct states must be described as part of the general pattern.
        Example: "Pattern includes two states: one with the lid open, and one with the lid closed."
        
        YOUR TASK:
        Write a **concise, general pattern** that includes:
        - The main objects and their key parts.
        - A summarized description of the object's static state, configuration, and position.
        - Important static variations (e.g., "lid open / lid closed").
        
        Output exactly one line in this format:
        <Frame {frame_number}>: [concise, general pattern summarizing all objects, static states, and key visual details/variations]
    '''
    
    return system_instruction, task_instruction


def _get_static_anomaly_prompt() -> tuple[str, str]:
    """Get prompts for static anomaly detection (no arrows)."""
    system_instruction = '''
        You are an anomaly detection expert for visual analysis. Your job is to compare a STATIC frame description set against a pattern and detect visual anomalies.
    '''
    
    task_instruction = '''
        INPUT:
        - A STATIC frame description set with 8 frames (Frame 1 to Frame 8), in time order
        - A pattern showing what NORMALLY appears visually in this scene
        
        WHAT IS ANOMALY IN STATIC DESCRIPTIONS:
        Anomaly means something VISUALLY STRANGE or UNEXPECTED compared to the pattern:
        - UNEXPECTED OBJECTS appearing (foreign objects, extra items)
        - MISSING OBJECTS that should be present
        - WRONG OBJECT APPEARANCE (different shape, color, size, texture than expected)
        - WRONG OBJECT STATE (open when should be closed, broken when should be intact, etc.)
        - WRONG SPATIAL ARRANGEMENT (objects in wrong positions or orientations)
        - ABNORMAL PARTS OR COMPONENTS (missing parts, damaged parts, wrong configuration)
        
        WHAT TO COMPARE:
        1. OBJECTS: Are the same objects present? Are unexpected objects appearing?
        2. APPEARANCE: Do objects look correct? (color, shape, size, texture, material)
        3. PARTS: Are all expected parts visible and intact?
        4. STATE: Are objects in the expected configuration? (open/closed, extended/compressed, upright/tilted)
        5. SPATIAL POSITION: Are objects where they should be?
        6. SURFACE DETAILS: Any unexpected markings, damage, or changes?
        
        YOUR TASK:
        Compare the visual details in the frame description set to the pattern:
        1. Check if all expected objects are present
        2. Check if object appearances match (colors, shapes, sizes)
        3. Check if object states/configurations are correct
        4. Check if spatial arrangements match
        5. Look for unexpected foreign objects or missing expected objects
        6. If anything visually doesn't match → ANOMALY
        
        SCORING GUIDE:
        - 0.0-0.2 = No anomaly (all visual details match the pattern)
        - 0.3-0.5 = Minor visual anomaly (small differences in appearance or position)
        - 0.6-0.8 = Moderate visual anomaly (clear differences: wrong colors, missing parts, unexpected objects)
        - 0.9-1.0 = Severe visual anomaly (completely wrong object, major damage, or multiple anomalies)
        
        OUTPUT FORMAT:
        Brief visual reasoning, then end with:
        Anomaly Score: <your score>
        
        Do not write anything after the score line.
    '''
    
    return system_instruction, task_instruction


def qwen_question(n):
    # identical to qwen_baseline.py except the frame count
    return f'''You are an expert in anomaly detection. Given the {n} frames of the video, look at the videos and determine whether the industrial object's appearance, motion, interaction with surroundings is normal or abnormal.
ANOMALY SCORING GUIDE: strictly only output in this format:
anomaly score: <a score between 0 (highly normal) and 1(highly anomaly)>
Reasoning: <a precise, logical, and evidence-based explanation describing why the behavior is considered normal or anomaly>
'''
