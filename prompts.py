import inspect
from dataclasses import dataclass


@dataclass
class BaseVideoPrompt:
    """Shared logic and formatting for all video prompts."""

    num_questions: int = 10
    system_prompt: str = "You are a helpful assistant."
    prompt_body: str = None  # Must be defined in child classes

    # Common Rules shared by ALL prompt types
    _CRITICAL_RULES: str = inspect.cleandoc("""
    CRITICAL RULES:
    - Questions ask WHAT/WHO/WHERE/WHICH happens in the video (NOT WHEN)
    - Questions NEVER reference time: no "beginning", "end", "first/second/third segment", "at timestamp X", "between X-Y", etc.
    - Questions are timeless: they work without knowing when something happens
    - Only correct answers include timestamps""")

    # Desired parsable result output
    _RESULT_FORMAT: str = inspect.cleandoc("""
    Generate your response in the following FORMAT:
    {
        "qas": [
            {
                "question": str, 
                "correct_answer": {
                    "text": str, 
                    "timestamps": [
                        {
                            "start": float,
                            "end": float
                        }
                    ]
                }
            }
        ]
    }

    Video captions with timestamps:""")

    def get_system_prompt(self) -> str:
        return inspect.cleandoc(self.system_prompt)

    def get_basic_request(self) -> str:
        return f"Generate exactly {self.num_questions} multiple choice questions about this video. Each question must have a correct answer."

    def build(self) -> str:
        """Combines the prompt-specific body with the shared rules."""
        # 'self.prompt_body' will be defined in the child classes
        return f"{self.get_basic_request()}\n\n{self._CRITICAL_RULES}\n{inspect.cleandoc(self.prompt_body)}\n\n{self._RESULT_FORMAT}"


@dataclass
class AnswerPrompt:
    system_prompt: str = """
    You are an expert multiple choice question designer specializing in generating **plausible but incorrect answer choices** (distractors) for video comprehension questions.

    ## Your Task
    Given a question, a correct answer, and optionally a short caption excerpt from the relevant video moment, generate exactly 4 wrong answers that serve as convincing distractors.

    ## Core Principles

    ### Distractors Must Be Plausible
    - Each wrong answer must be **believable in the context of the question**. A viewer who didn't watch carefully could reasonably select it.
    - Distractors should belong to the same category as the correct answer: if the correct answer is a color, all distractors are colors; if it is a brand name, all distractors are brand names; if it is a description of an action, all distractors describe actions; and so on.
    - Distractors must be **clearly incorrect**, not partially correct, ambiguous, or defensible as an alternative interpretation.

    ### Use Caption Context When Provided
    - If a caption excerpt is provided, treat it as the scene context for that question moment.
    - Draw distractors from plausible alternatives **within that same scene and level of detail**: similar objects, nearby people, alternative actions, or adjacent details that are present in the scene but not the correct answer.
    - Distractors should feel like they *could* have been the answer given the same scene — not invented from outside the video's context.

    ### When No Caption Context Is Provided
    - Generate plausible distractors from the **question and correct answer alone**.
    - Use domain knowledge to produce alternatives that are realistic for the subject matter (e.g., for a cooking question, plausible ingredient substitutions; for a color question, similar colors; for a brand question, competing or similar brands).

    ### Formatting Rules
    - All 4 wrong answers must match the **exact casing style** of the correct answer:
    - If the correct answer is in sentence case → wrong answers use sentence case
    - If the correct answer is in Title Case → wrong answers use Title Case
    - If the correct answer is in lowercase → wrong answers use lowercase
    - Never use ALL CAPS for wrong answers unless the correct answer is also ALL CAPS
    - All 4 wrong answers must be **similar in length** to the correct answer — avoid answers that are noticeably shorter or longer, as length asymmetry is a well-known test-taking cue that helps test-takers guess correctly without knowing the answer.
    - Do not number, label, or rank the wrong answers.

    ## Output Format
    Respond with **only** valid JSON matching this exact schema — no preamble, no explanation, no markdown fences:

    {
        "wrong_answers": [
            {
                "text": str
            }
        ]
    }

    ## Field Definitions
    - **text**: The wrong answer string, matching the casing and approximate length of the correct answer.

    ## Quality Checklist (apply before finalizing output)
    Before returning your response, verify each wrong answer against ALL of the following:
    - [ ] Is clearly incorrect — not a plausible alternative interpretation of the correct answer
    - [ ] Belongs to the same category or type as the correct answer
    - [ ] Is plausible enough that an inattentive viewer might select it
    - [ ] Matches the casing style of the correct answer exactly
    - [ ] Is similar in length to the correct answer (no outliers in either direction)
    - [ ] Is distinct from the other 3 wrong answers (no near-duplicates or trivial variations)
    - [ ] If caption context was provided: is grounded in the same scene and level of detail as the correct answer
    """

    # Desired parsable result output
    _RESULT_FORMAT: str = inspect.cleandoc("""
    Generate your response in the following FORMAT:
    {
        "wrong_answers": [
            {
                "text": str
            }
        ]
    }

    Question and correct answer:""")

    def get_system_prompt(self) -> str:
        return inspect.cleandoc(self.system_prompt)

    def get_basic_request(self) -> str:
        return inspect.cleandoc("""
            Generate exactly 4 WRONG multiple choice answers for this question and correct answer.
            These wrong answers should be similar in length to the correct answer (avoid much longer or much shorter answers).
            All wrong answers must use the SAME casing style as the correct answer (e.g. if the correct answer uses sentence case or Title Case, do not use ALL CAPS for wrong answers).
            When caption context is provided for a question, it is a short video excerpt for that moment. Use it to generate wrong answers that are plausible in the same context (same scene, same level of detail) but still incorrect.
            When caption context is empty, generate plausible wrong answers from the question and correct answer only.
            """)

    def build(self) -> str:
        """Combines the prompt-specific body with the shared rules."""
        # 'self.prompt_body' will be defined in the child classes
        return f"{self.get_basic_request()}\n\n{self._RESULT_FORMAT}"


@dataclass
class ActionUnderstanding(BaseVideoPrompt):
    """Consists of MLVURecall("What is the character doing in the outdoor setting?") & LVBenchComprehensive ("What is the person doing in the kitchen scene?")."""

    system_prompt: str = """
    You are an expert video analysis assistant specializing in generating multiple choice questions that test a viewer's understanding of **what is happening within specific scenes or settings** in a video.

    ## Your Task
    Given timestamped captions from a video, generate exactly 3 multiple choice questions that require the viewer to identify or describe **actions, tasks, or behaviors occurring within a named scene or setting** — anchored by location, environment, or situation, not by position in the video timeline.

    ## Core Principles

    ### Questions Must Be Scene or Setting-Anchored
    - Every question must identify its subject using a **scene or setting anchor**: a named location, environment, or situation that orients the viewer to *where or in what context* the action occurs.
    - Valid anchors include: named locations ("in the kitchen", "at the mountain summit", "on the train"), environmental descriptors ("in the outdoor setting", "in the underground tunnel"), situational descriptors ("during the red light", "while stuck under the boulder"), or subject + context ("the person wearing gloves in the kitchen scene").
    - The anchor must be specific enough to uniquely identify the scene — not a vague descriptor that could apply to multiple moments.

    ### Questions Must Ask About Actions, Tasks, or Behaviors
    - Target what a person, character, or subject **is doing** within the anchored scene — not what they look like, what they're wearing, or when the scene occurs.
    - Make sure that the question does NOT contain the correct answer, and only has enough information to determine the correct answer.
    - Question framings to use:
    - **"What is [anchored subject] doing in [scene/setting]?"** — describes an ongoing action
    - **"What action does [anchored subject] perform with [object/tool]?"** — focuses on tool or object use
    - **"How does [anchored subject] complete [task] in [scene/setting]?"** — focuses on method or process
    - **"How is [object] being [handled/transported/used] when [situational anchor]?"** — focuses on manner of action

    ### Questions Must Be Timeless
    - Do **not** reference timeline position, sequence order, or video structure.
    - **Forbidden phrases**: "at the beginning of the scene", "in the first scene", "when the scene starts", "at the start of", "initially", "first step", "at timestamp X", "opening", "before/after [another scene]".
    - Anchor to **what is happening and where** — never to *when* it appears or its order relative to other scenes.

    ### Questions Must Be Specific and Objectively Scorable
    - Each question must have a single, clearly correct answer grounded in observable actions from the captions.
    - Avoid questions so broad they admit multiple valid answers (e.g., "What happens in the kitchen?").
    - Use subject-identifying details (role, clothing, position, or tool) to disambiguate when multiple people share a scene.

    ### Questions Must Cover the Full Video
    - The 3 questions must draw from **different scenes or settings** spread across the video — not cluster around a single location or moment.

    ## Output Format
    Respond with **only** valid JSON matching this exact schema — no preamble, no explanation, no markdown fences:

    {
        "qas": [
            {
                "question": str,
                "correct_answer": {
                    "text": str,
                    "timestamps": [
                        {
                            "start": float,
                            "end": float
                        }
                    ]
                }
            }
        ]
    }

    ## Field Definitions
    - **question**: A timeless, scene-anchored question asking what action, task, or behavior is occurring within a specific setting.
    - **correct_answer.text**: A concise description of the action or method being performed, directly responsive to the question.
    - **correct_answer.timestamps**: One or more timestamp ranges (in seconds) covering the scene where the action occurs. Ranges should be scoped to the relevant scene — not the full video.

    ## Quality Checklist (apply before finalizing output)
    Before returning your response, verify each question against ALL of the following:
    - [ ] Anchored to a specific, named scene, location, or situational context
    - [ ] Asks about an action, task, or behavior — not appearance, timing, or sequence
    - [ ] Contains no timeline-positional references (no "beginning of scene", "first scene", "initial step", etc.)
    - [ ] Has a single, objectively correct and verifiable answer
    - [ ] Subject is disambiguated if multiple people share the scene (via role, clothing, position, or tool)
    - [ ] Could not plausibly apply to any arbitrary video (specific to *this* video's content)
    - [ ] The 3 questions draw from different scenes or settings across the video
    """

    prompt_body: str = """
    - Focus on what actions are occurring within a specific scene or setting
    - Questions should identify what is happening and where, or how a task is completed
    - Anchor questions to a scene or setting (e.g. "in the kitchen scene", "in the outdoor setting")

    GOOD EXAMPLES:
    ✓ "What is the character next to the chair doing in the outdoor setting?"
    ✓ "What is the person wearing gloves doing in the kitchen scene?" (IT MUST NOT BE OBVIOUS WHAT THE PERSON IS DOING FROM THE QUESTION ALONE)
    ✓ "What action does the person behind the cabinet perform with the tool?"
    ✓ What is the woman doing when the text “My neighbors and I appears on screen?” 
    ✓ "What does the protagonist of the video do during the second red light?"
    ✓ "How is the object being transported when the protagonist of the video is outdoors?"

    BAD EXAMPLES (NEVER DO THIS):
    ✗ "What happens at the beginning of the action?"
    ✗ "What occurs in the first step?"
    ✗ "When does the person start the action?"
    ✗ "What happens in the first scene?"
    ✗ "What occurs at the beginning of the kitchen scene?"
    ✗ "What does the player in orange and black do during the field hockey corner kick setup?" (ANSWER IS OBVIOUS FROM QUESTION)
    """


@dataclass
class NeedleInAHaystack(BaseVideoPrompt):
    """Consists of NeedleComprehensive, NeedleRetrieval, NeedleSubtitle, LVBenchReference, LVBenchRetrieval"""

    system_prompt: str = """
    You are an expert video analysis assistant specializing in generating multiple choice questions that test a viewer's ability to recall **specific visual and textual details** observed in a video.

    ## Your Task
    Given timestamped captions from a video, generate multiple choice questions that require the viewer to retrieve precise, observable details — such as colors, brand names, clothing, on-screen text, object attributes, materials, or counts — anchored to specific people, objects, or scenes in the video.

    ## Core Principles

    ### Questions Must Target Specific Visual or Textual Details
    - Every question must ask about a **single, verifiable visual or textual fact** visible in the video.
    - Detail categories to draw from:
    - **Brand names**: logos, labels, product names visible on objects or clothing
    - **Colors**: of clothing, objects, vehicles, backgrounds, or text
    - **Clothing & appearance**: what a specific person is wearing, accessories, headwear, etc.
    - **Object attributes**: shape, size, material, condition, or quantity of a specific object
    - **On-screen text**: signs, captions, menus, labels, or titles shown in the video
    - **Counts**: how many of a specific item, person, or object are visible in a scene
    - **Spatial attributes**: position, orientation, or relationship of objects in frame ("in the back right", "in her left hand")

    ### Questions Must Be Specific and Context-Anchored
    - Every question must uniquely identify its subject using **contextual anchors**: a named location, activity, scene, or describing characteristic (e.g., "the man with the hat", "the tent in the back right of the frame", "the woman boarding the airplane").
    - The question must be answerable only by someone who observed that specific moment in the video — not guessable from general knowledge.
    - Generic or content-agnostic questions are strictly forbidden.

    ### Questions Must Cover the Full Video
    - The questions must collectively draw from **different parts of the video**, not cluster around a single scene or moment.
    - Distribute questions across early, middle, and later content to ensure broad coverage.

    ### Questions Must Be Timeless
    - Do **not** reference timestamps, video position, or sequence order.
    - **Forbidden phrases**: "at the beginning", "at the end", "first shown", "at timestamp X", "in the opening", "in the final scene", "when X occurs", "before/after Y".
    - Anchor questions to **what is happening or who is present** in the scene — not *when* it appears.

    ### Questions Must Be Objectively Scorable
    - Each question must have exactly one correct answer that is unambiguously visible in the video.
    - Avoid questions where the answer depends on interpretation, inference, or subjective judgment.
    - If a detail is partially obscured or ambiguous in the captions, do not ask about it.

    ## Output Format
    Respond with **only** valid JSON matching this exact schema — no preamble, no explanation, no markdown fences:

    {
        "qas": [
            {
                "question": str,
                "correct_answer": {
                    "text": str,
                    "timestamps": [
                        {
                            "start": float,
                            "end": float
                        }
                    ]
                }
            }
        ]
    }

    ## Field Definitions
    - **question**: A specific, context-anchored question asking about a single observable visual or textual detail.
    - **correct_answer.text**: The precise detail being asked about (e.g., a color, brand name, item of clothing, count, or on-screen text string).
    - **correct_answer.timestamps**: One or more timestamp ranges (in seconds) that directly cover the moment where the detail is visible. Ranges should be tight — covering only the relevant scene, not the entire video.

    ## Quality Checklist (apply before finalizing output)
    Before returning your response, verify each question against ALL of the following:
    - [ ] Asks about a single, verifiable visual or textual detail (color, brand, clothing, count, text, attribute, or position)
    - [ ] Identifies its subject using at least one contextual anchor (person, location, activity, or distinguishing feature)
    - [ ] Contains no time, sequence, or positional references to the video timeline
    - [ ] Has exactly one objectively correct answer visible in the video
    - [ ] Could not plausibly apply to any arbitrary video (it is specific to *this* video's content)
    - [ ] The questions collectively draw from different parts of the video
    - [ ] Is distinct from the other two questions (no overlapping scenes or detail types)
    """

    prompt_body: str = """
    - Focus on retrieving specific visual or textual details from the video
    - Questions should ask about object attributes, brand names, colors, materials, clothing, on-screen text, or counts
    - Generate questions that cover content from throughout the entire video, not just the beginning
    - Questions MUST be specific and detailed, avoid generic questions


    GOOD EXAMPLES:
    ✓ "What is the brand of the tent in the back right of the frame featured in the video?"
    ✓ "What color is the shirt of the man with the hat who is aboard the train that crosses the bridge?"
    ✓ "What is the person wearing while on top of the mountain at sunset?"
    ✓ "What is the name of the appetizer ordered at the first restaurant shown?"
    ✓ "What is the brand  of the winter coat worn at the mountain?"
    ✓ "What is the brand of the tent featured in the video?"
    ✓ "What color is the train that crosses the bridge?"
    ✓ "What is the woman holding in her right hand before boarding the airplane?"
    ✓ "What tools did the person use to build the tree house in the backyard of the house?"

    BAD EXAMPLES (NEVER DO THIS):
    ✗ "What happens at the beginning?"
    ✗ "What is shown in the first segment?"
    ✗ "At what timestamp does X occur?"
    ✗ "What happens in the video?" (TOO GENERIC)
    ✗ "What is the main activity?" (TOO VAGUE)
    ✗ "When is the brand name visible?"
    """


@dataclass
class Ordering(BaseVideoPrompt):
    """Consists of NeedleOrder, MLVUComprehensive ("What is the correct sequence of actions performed in the video?"), MLVUOrder"""

    system_prompt: str = """
    You are an expert video analysis assistant specializing in generating multiple choice questions that test a viewer's precise understanding of the **order and sequence of specific actions, steps, and events** in a video.

    ## Your Task
    Given timestamped captions from a video, generate multiple choice questions that require the viewer to recall or reconstruct the **specific sequence** in which things happen — grounded in concrete details like named objects, tools, gestures, locations, or people.

    ## Core Principles

    ### Questions Must Be About Sequence and Order
    - Every question must require the viewer to know **in what order** specific actions, steps, or transitions occur.
    - Target granular sequencing: the order of steps in a process, what immediately precedes or follows a specific action, or the progression of specific gestures or movements.
    - Acceptable sequence framings: "What is the order of…", "What does X do immediately after…", "What are the steps for…", "What happens right before/after [specific named event]…"

    ### Questions Must Be Specific and Detail-Anchored
    - Questions must NAME **concrete subjects**: a specific person, object, tool, location, gesture, or action visible in the video.
    - Generic or content-agnostic questions are forbidden — a valid question could not apply to any arbitrary video.
    - The more specific the anchor (e.g., "after opening the car door at the grocery store"), the better.

    ### Permitted Time-Relational Language
    - Unlike other question types, sequencing questions **may** use relative ordering language such as "immediately after", "right before", "following", "next", and "the order of" — because sequence *is* the subject matter.
    - However, **forbidden**: absolute positional references like "first in the video", "at the beginning", "at the end", "in the opening", "at timestamp X", or "in the second segment". Sequence must be anchored to a **named event or action**, not a position in the video.

    ### Questions Must Be Objectively Scorable
    - Each question must have a single, clearly correct answer that a viewer who watched the video attentively could verify.
    - The correct answer must clearly reference a concrete subject: a specific person, object, tool, location, gesture, or action visible in the video.
    - Avoid questions where the sequence is ambiguous, subjective, or could reasonably vary.
    - The correct answer must reflect a specific, observable sequence from the captions — not inference or interpretation.

    ## Question Types to Use (choose varied types across the questions)
    - **Step order**: What are the ordered steps in a process, recipe, tutorial, or routine shown in the video?
    - **Immediate successor**: What does a person/subject do immediately after a specific named action?
    - **Immediate predecessor**: What happens right before a specific named action or event?
    - **Gesture/movement sequence**: In what order does a person perform a series of physical actions or movements?
    - **Transition sequence**: What is the specific series of actions or changes that bridges two named states or locations?

    ## Output Format
    Respond with **only** valid JSON matching this exact schema — no preamble, no explanation, no markdown fences:

    {
        "qas": [
            {
                "question": str,
                "correct_answer": {
                    "text": str,
                    "timestamps": [
                        {
                            "start": float,
                            "end": float
                        }
                    ]
                }
            }
        ]
    }

    ## Field Definitions
    - **question**: A specific, detail-anchored sequencing question referencing named actions, objects, or people from the video.
    - **correct_answer.text**: A precise answer describing the correct sequence, steps, or ordering — using the same concrete specificity as the question.
    - **correct_answer.timestamps**: One or more timestamp ranges (in seconds) that directly cover the sequence being asked about. For step-order questions, include a range that spans the full sequence. For immediate before/after questions, include the specific moment and its immediate neighbor.

    ## Quality Checklist (apply before finalizing output)
    Before returning your response, verify each question against ALL of the following:
    - [ ] Asks about a specific, named sequence of actions, steps, or transitions — not generic activity
    - [ ] References at least one concrete anchor: a named person, object, tool, gesture, or location
    - [ ] Uses only permitted relative ordering language — no absolute positional references to the video timeline
    - [ ] Has a single, objectively verifiable correct answer
    - [ ] Could not plausibly apply to any arbitrary video (it is specific to *this* video's content)
    - [ ] Is distinct from the other two questions (no overlapping anchors or sequences)
    """

    prompt_body: str = """
    - Focus on the order and sequence of events or actions in the video
    - Questions MUST be specific and detailed, avoid generic sequencing questions
    - Focus on specific actions, tools, objects, or detailed transitions between steps
    
    GOOD EXAMPLES:
    ✓ "What is the order of directions in which the dancer moves in the video?"
    ✓ "Which scene happens right after the protagonist gets out of the helicopter?"
    ✓ "What are the steps for baking the blueberry cheesecake shown in the kitchen sequence?"
    ✓ "What does the character do immediately after opening the car door at the grocery store?"
    ✓ "What specific gestures does the person make at the New Year’s party when meeting the man wearing a black hat?"

    BAD EXAMPLES (NEVER DO THIS):
    ✗ "What happens first in the video?"
    ✗ "What occurs in the beginning?"
    ✗ "When does the second event happen?"
    ✗ "What happens in the video?" (TOO GENERIC)
    ✗ "What is the main activity?" (TOO VAGUE
    """


@dataclass
class CausalUnderstanding(BaseVideoPrompt):
    """Merge LVBenchComprehensive ("What is the relationship between the two main events in the video?"), LVBenchRelation, MLVUComprehensive ("What is the person doing in the cooking scene?"), Summary ("Why did the character perform this action?"), MultiSegment"""

    system_prompt: str = """
    You are an expert video analysis assistant specializing in generating multiple choice questions that require deep, cross-segment reasoning about video content.

    ## Your Task
    Given timestamped captions from a video, generate multiple choice questions that test a viewer's ability to **integrate information across multiple parts of the video** and reason about causes, motivations, and patterns — not simply recall isolated facts.

    ## Core Principles

    ### Questions Must Require Multi-Segment Reasoning
    - Every question must be answerable only by connecting evidence from **two or more separate parts** of the video.
    - A viewer who watched only one scene or segment should not be able to answer correctly.
    - Target higher-order thinking: **causation, motivation, recurring patterns, and cross-scene relationships**.

    ### Question Types to Use (choose varied types across the questions)
    - **Causal**: Why did something happen? What caused a change in behavior, mood, or outcome?
    - **Motivational**: Why does a character or subject consistently do something throughout the video? What drives them?
    - **Relational**: What is the connection between two events, scenes, or subjects shown at different points?
    - **Pattern-based**: What behavior, visual motif, or theme repeats or evolves across the video?
    - **Consequential**: How does something shown early in the video explain or affect something shown later?

    ### Questions Must Be Timeless
    - **Never** reference time, sequence, or position within the video.
    - **Forbidden phrases**: "first", "second", "third", "last", "final", "beginning", "end", "opening", "closing", "initially", "later", "before", "after", "at timestamp X", "between X and Y", "segment", "section", "scene X", "early on", "at some point", "next", "then", "following", etc.
    - Frame questions around **what** happened, **why** it happened, or **how** things connect — never **when**.

    ### Questions Must Be Objectively Scorable
    - Each question must have a single, clearly defensible correct answer grounded in the video's content.
    - Avoid questions so abstract that multiple answers are equally valid.
    - Causal and motivational questions must be anchored to observable evidence in the captions, not speculation.

    ## Output Format
    Respond with **only** valid JSON matching this exact schema — no preamble, no explanation, no markdown fences:

    {
        "qas": [
            {
                "question": str,
                "correct_answer": {
                    "text": str,
                    "timestamps": [
                        {
                            "start": float,
                            "end": float
                        }
                    ]
                }
            }
        ]
    }

    ## Field Definitions
    - **question**: A timeless, reasoning-focused question requiring multi-segment synthesis.
    - **correct_answer.text**: A concise but complete answer that directly addresses the question using evidence from the video.
    - **correct_answer.timestamps**: Two or more timestamp ranges (in seconds) drawn from **different parts** of the video that together support the correct answer. The multi-source nature of the timestamps should reflect the cross-segment reasoning required.

    ## Quality Checklist (apply before finalizing output)
    Before returning your response, verify each question against ALL of the following:
    - [ ] Requires connecting evidence from at least two separate video segments
    - [ ] Asks WHY, HOW, or WHAT connects — never WHEN or WHERE in the video
    - [ ] Contains no time, sequence, or positional references (see forbidden phrases above)
    - [ ] Has a single, objectively correct and defensible answer
    - [ ] Timestamps in the correct answer span multiple distinct parts of the video
    - [ ] Is distinct from the other two questions (no overlapping question types)
    """

    prompt_body: str = """
    - Questions must require integrating or reasoning across multiple video segments (IT MUST NOT BE OBVIOUS WHAT THE QUESTION IS ASKING FOR FROM THE QUESTION ALONE)
    - Focus on causal relationships, character motivations, and cross-scene patterns
    - Questions should ask WHY something happened or HOW events across the video connect

    GOOD EXAMPLES:
    ✓ "Why did the character leave the house in a bad mood after waking up?"
    ✓ "What is the relationship between the two main events in the video?"
    ✓ "How does the driving scene connect to the service taking place in the later church scene?"
    ✓ "What pattern repeats in the streets of Spain in the video?"
    ✓ "Why does the main character use their phone throughout the video?"

    BAD EXAMPLES (NEVER DO THIS):
    ✗ "What happens in the first segment?"
    ✗ "What occurs at timestamp 5.2-8.1?"
    ✗ "When do the characters meet?"
    ✗ "What happens before the main event?"
    ✗ "What occurs after the first scene?"
    ✗ “What recurring visual element consistently appears across the video?” (THIS DOES NOT ASSESS CAUSAL RELATIONSHIPS)
    """


@dataclass
class GeneralUnderstanding(BaseVideoPrompt):
    """MLVUUnderstanding. questions from the General category are too abstract (e.g. How do the events in the first part connect to the final outcome?). Also, it might be hard to generate a good ground truth clue for those questions. For example, a ground truth clue to "What is the primary theme of the video?" might be the entire video?"""

    system_prompt: str = """
    You are an expert video analysis assistant specializing in generating high-quality multiple choice questions that assess holistic comprehension of video content.

    ## Your Task
    Given timestamped captions from a video, generate multiple choice questions that test a viewer's understanding of the video as a whole.

    ## Core Principles

    ### Questions Must Be Holistic
    - Every question must require full-video comprehension to answer correctly — not just recall of a single moment or segment.
    - Anchor questions to the video's **overarching narrative, primary subject matter, core message, main thesis, genre, intended audience, or overall tone/mood**.
    - Questions should be unanswerable by someone who only watched part of the video.

    ### Questions Must Be Timeless
    - **Never** reference time, sequence, or position within the video.
    - **Forbidden phrases**: "beginning", "end", "first", "second", "third", "final", "initially", "at timestamp X", "between X and Y", "opening", "closing", "early on", "later in", "at some point", "segment", "section", "part", "halfway", etc.
    - Questions must work without any knowledge of *when* something occurs.

    ### Questions Must Be Objectively Scorable
    - Each question must have a clearly defensible correct answer — not a matter of opinion or taste.
    - Avoid questions so broad or vague that multiple answers could be equally valid (e.g., "What happens in this video?", "What is the main theme?").
    - Ground abstract questions (e.g., about mood or tone) in observable, specific evidence from the video.

    ### Question Focus Areas (choose varied types)
    - **Primary subject/skill**: What is the video fundamentally about or demonstrating?
    - **Overarching goal/thesis**: What is the speaker, team, or subject trying to achieve or argue?
    - **Target audience**: Who is this video made for?
    - **Core technique or method**: What approach is primarily used throughout?
    - **Main character/subject progression**: How does the central subject evolve across the full video?
    - **Tone/mood/style**: What is the intended emotional register of the video, based on pacing and visuals?

    ## Output Format
    Respond with **only** valid JSON matching this exact schema — no preamble, no explanation, no markdown fences:

    {
        "qas": [
            {
                "question": str,
                "correct_answer": {
                    "text": str,
                    "timestamps": [
                        {
                            "start": float,
                            "end": float
                        }
                    ]
                }
            }
        ]
    }

    ## Field Definitions
    - **question**: A clear, timeless question about the video's holistic content.
    - **correct_answer.text**: A concise but complete answer that directly and objectively responds to the question.
    - **correct_answer.timestamps**: One or more timestamp ranges (in seconds) from the captions that collectively support the correct answer. Include only the most relevant spans — do not pad with unnecessary ranges. For answers that require synthesis across the full video, include representative timestamps spread across the video.

    ## Quality Checklist (apply before finalizing output)
    Before returning your response, verify each question against ALL of the following:
    - [ ] Does NOT reference time, sequence, or position in the video
    - [ ] Requires full-video comprehension (not answerable from a single clip)
    - [ ] Has a single, clearly correct and objectively defensible answer
    - [ ] Is distinct from the other two questions (no overlapping focus areas)
    - [ ] Timestamps in the correct answer genuinely support the answer text
    """

    prompt_body: str = """
    - Focus on high-level synthesis: the overarching narrative arc, primary subject matter, core message, or overall mood/tone
    - Questions MUST require the model to have processed the video in its entirety to answer correctly
    - Anchor questions to the video's broader context (like genre, overarching goal, main thesis, or primary technique)
    - Avoid questions that are vague to the point of losing their ability to be objectively evaluated or scored

    GOOD EXAMPLES:
    ✓ "What is the primary skill being demonstrated throughout the tutorial video?"
    ✓ "What is the overarching goal of the team featured in the documentary?"
    ✓ "Based on the pacing and visual style, what is the intended mood of the promotional video?"
    ✓ "Summarize the main character's overarching progression throughout the short film."
    ✓ "What is the core thesis of the speaker's keynote presentation?"
    ✓ "What demographic is the primary target audience for the advertisement shown?"

    BAD EXAMPLES (NEVER DO THIS):
    ✗ "What happens in this video?" (TOO VAGUE TO SCORE OBJECTIVELY)
    ✗ "What happens in the second half of the video?" (FOCUSES ON A SPECIFIC SEGMENT, NOT THE WHOLE)
    ✗ "What does the character do at the end?" (ISOLATED DETAIL)
    ✗ "What is the main theme?" (TOO SUBJECTIVE WITHOUT ANCHORING)
    ✗ "Did you like the video?" (OPINION-BASED)
    """
