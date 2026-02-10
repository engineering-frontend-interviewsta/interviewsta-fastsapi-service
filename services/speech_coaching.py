"""
Speech & English Coaching Service
Provides dynamic coaching text and meaning based on score ranges for each speech metric.
"""
import random


def force_randomize_speech(speech):
    """
    Force randomize speech scores with a base value and independent noise.
    
    Args:
        speech: dict with 'grammar', 'fluency', 'fillers', 'clarity' scores (or None)
        
    Returns:
        dict: Randomized speech scores, or original if speech is None/empty
    """
    if not speech:
        return speech

    base = random.randint(55, 85)

    return {
        "grammar": max(0, min(100, base + random.randint(-15, 15))),
        "fluency": max(0, min(100, base + random.randint(-12, 12))),
        "fillers": max(0, min(100, base + random.randint(-18, 18))),
        "clarity": max(0, min(100, base + random.randint(-10, 10))),
    }


def generate_speech_scores(soft):
    """
    Generate speech scores from soft skill metrics with independent noise channels.
    
    Args:
        soft: dict with keys 'confidence', 'nervousness', 'engagement', 'eye_contact'
        
    Returns:
        dict: Contains 'grammar', 'fluency', 'fillers', 'clarity' scores (0-100)
    """
    noiseA = random.randint(-18, 18)
    noiseB = random.randint(-14, 14)
    noiseC = random.randint(-20, 20)
    noiseD = random.randint(-16, 16)

    grammar = (
        0.45 * soft["confidence"] +
        0.30 * (100 - soft["nervousness"]) +
        0.25 * soft["eye_contact"] +
        noiseA
    )

    fluency = (
        0.50 * soft["engagement"] +
        0.30 * soft["confidence"] +
        0.20 * (100 - soft["nervousness"]) +
        noiseB
    )

    fillers = (
        0.65 * (100 - soft["nervousness"]) +
        0.20 * soft["confidence"] +
        0.15 * soft["engagement"] +
        noiseC
    )

    clarity = (
        0.40 * soft["confidence"] +
        0.35 * soft["engagement"] +
        0.25 * soft["eye_contact"] +
        noiseD
    )

    return {
        "grammar": max(0, min(100, round(grammar))),
        "fluency": max(0, min(100, round(fluency))),
        "fillers": max(0, min(100, round(fillers))),
        "clarity": max(0, min(100, round(clarity))),
    }


def get_score_category(score):
    """
    Categorize score into weak, moderate, or strong.
    
    Args:
        score: Integer score between 0-100
        
    Returns:
        str: 'weak', 'moderate', or 'strong'
    """
    if score < 70:
        return 'weak'
    elif score < 85:
        return 'moderate'
    else:
        return 'strong'


def get_grammar_coaching(score):
    """
    Get grammar coaching text based on score.
    
    Args:
        score: Integer score between 0-100
        
    Returns:
        dict: Contains 'meaning' and 'coaching' text
    """
    category = get_score_category(score)
    
    if category == 'weak':
        meaning = "Your grammar needs significant improvement. Frequent errors can undermine your credibility and make it harder for interviewers to focus on your ideas."
        coaching = (
            "Grammar is the foundation of professional communication, and improving it will significantly enhance how interviewers perceive your competence and attention to detail. "
            "Start by focusing on subject-verb agreement, which is one of the most common errors in spoken English. Practice speaking in complete sentences and avoid run-on sentences that can confuse your message. "
            "Pay special attention to verb tenses, especially when discussing past experiences or future plans. Consider recording yourself speaking and listening back to identify patterns in your grammatical mistakes. "
            "Reading aloud from well-written articles or books can help you internalize correct sentence structures. Additionally, practice explaining technical concepts clearly, as this often reveals grammar weaknesses. "
            "Remember that good grammar isn't just about correctness—it's about making your ideas clear and accessible. When you speak with proper grammar, interviewers can focus on what you're saying rather than how you're saying it. "
            "This builds trust and demonstrates that you pay attention to details, which is crucial in professional settings. With consistent practice, you'll notice your grammar improving naturally over time."
        )
    elif category == 'moderate':
        meaning = "Your grammar is acceptable but has room for improvement. Occasional errors may distract from your message."
        coaching = (
            "You have a solid foundation in grammar, but refining it will help you communicate more effectively and professionally. "
            "Focus on eliminating the occasional errors that slip through, particularly with complex sentence structures or when you're speaking under pressure. "
            "Practice using varied sentence structures to make your speech more engaging and professional. Pay attention to prepositions and articles, as these small words can significantly impact clarity. "
            "When discussing technical topics, ensure you're using precise terminology and correct grammatical forms. Consider practicing with a focus on consistency—maintaining proper grammar throughout longer responses. "
            "Recording yourself during practice interviews can help you identify specific areas where errors tend to occur. Remember that moderate grammar skills can be elevated to strong with targeted practice. "
            "The goal is to make grammar feel natural and automatic, so you can focus entirely on your content during interviews. With continued attention to detail, you'll develop the fluency that makes your communication seamless and professional."
        )
    else:  # strong
        meaning = "Your grammar is strong and professional. You communicate clearly and effectively."
        coaching = (
            "Congratulations on maintaining strong grammar skills! Your ability to communicate with grammatical precision significantly enhances your professional presence. "
            "Continue to maintain this standard by staying mindful of grammar even in casual conversations, as consistency builds habits. Consider challenging yourself with more complex sentence structures and advanced grammatical concepts. "
            "You can also help others improve their grammar, which reinforces your own understanding. Remember that strong grammar is an ongoing commitment—even native speakers benefit from continued attention to language. "
            "Your grammatical competence allows interviewers to focus entirely on your ideas and expertise rather than being distracted by language errors. This creates a more positive interview experience and demonstrates your attention to detail. "
            "Keep up the excellent work, and consider using your strong grammar skills to enhance other aspects of communication, such as clarity and precision in technical explanations."
        )
    
    return {
        'meaning': meaning,
        'coaching': coaching
    }


def get_fluency_coaching(score):
    """
    Get fluency coaching text based on score.
    
    Args:
        score: Integer score between 0-100
        
    Returns:
        dict: Contains 'meaning' and 'coaching' text
    """
    category = get_score_category(score)
    
    if category == 'weak':
        meaning = "Your speech fluency needs significant improvement. Frequent pauses and hesitations can make you appear uncertain."
        coaching = (
            "Fluency is about speaking smoothly and confidently without excessive pauses or hesitations, and improving it will make you appear more prepared and self-assured. "
            "Start by practicing speaking about familiar topics without stopping to think—this builds the muscle memory of continuous speech. Focus on reducing filler words like 'um' and 'uh' by pausing silently instead when you need a moment. "
            "Practice speaking at a moderate pace rather than rushing, as this actually improves fluency by giving your brain time to process. Try recording yourself speaking for two minutes without stopping, then gradually increase the duration. "
            "Reading aloud regularly can help train your mouth and brain to work together more smoothly. When practicing, focus on maintaining a steady rhythm rather than perfect accuracy—fluency comes from flow. "
            "Remember that some pauses are natural and even professional when used strategically, but excessive hesitation suggests uncertainty. Building fluency requires consistent practice, so commit to speaking exercises daily. "
            "As your fluency improves, you'll notice that interviewers respond more positively because smooth speech signals confidence and preparation. This creates a better overall impression and allows your ideas to shine through clearly."
        )
    elif category == 'moderate':
        meaning = "Your fluency is decent but could be smoother. Some hesitations may interrupt your flow."
        coaching = (
            "You have a good foundation in fluency, but refining it will help you speak more confidently and professionally. "
            "Focus on reducing the occasional hesitations that interrupt your natural flow, particularly when transitioning between ideas or explaining complex concepts. "
            "Practice speaking in longer, connected sentences rather than short, choppy phrases. Work on maintaining your pace even when you encounter a challenging topic or question. "
            "Consider practicing with a focus on smooth transitions between thoughts, using connecting words and phrases naturally. Recording yourself can help you identify specific moments where fluency breaks down. "
            "Remember that moderate fluency can be elevated through consistent practice and mindfulness. The goal is to make speaking feel effortless, so you can focus on your message rather than your delivery. "
            "As you improve, you'll notice that smoother speech makes you appear more confident and prepared, which positively impacts how interviewers perceive your competence. Continue practicing, and your fluency will become increasingly natural."
        )
    else:  # strong
        meaning = "Your speech is fluent and smooth. You communicate with confidence and ease."
        coaching = (
            "Excellent work on maintaining strong fluency! Your ability to speak smoothly and confidently significantly enhances your professional presence and makes interviews more engaging. "
            "Continue to maintain this standard by staying mindful of your pace and flow even in casual conversations. Consider challenging yourself with more complex topics or longer explanations to keep your fluency skills sharp. "
            "You can use your strong fluency to enhance other aspects of communication, such as storytelling or technical explanations. Remember that maintaining fluency requires ongoing practice, even for strong speakers. "
            "Your smooth, confident speech allows interviewers to focus entirely on your ideas and expertise rather than being distracted by delivery issues. This creates a more positive interview experience and demonstrates your communication competence. "
            "Keep up the excellent work, and consider using your strong fluency as a foundation for further developing other communication skills like clarity and engagement."
        )
    
    return {
        'meaning': meaning,
        'coaching': coaching
    }


def get_fillers_coaching(score):
    """
    Get fillers coaching text based on score.
    
    Args:
        score: Integer score between 0-100
        
    Returns:
        dict: Contains 'meaning' and 'coaching' text
    """
    category = get_score_category(score)
    
    if category == 'weak':
        meaning = "You use too many filler words. Excessive 'um', 'uh', 'like', and 'you know' can undermine your credibility."
        coaching = (
            "Filler words are natural, but excessive use can make you appear uncertain, unprepared, or less professional. Reducing them will significantly improve how interviewers perceive your confidence and competence. "
            "Start by becoming aware of your filler word patterns—record yourself speaking and count how many times you use 'um', 'uh', 'like', 'you know', or similar words. Practice replacing these with brief, silent pauses instead. "
            "When you feel the urge to use a filler word, take a small breath or pause for half a second—this actually makes you appear more thoughtful rather than uncertain. Practice speaking slowly enough that you can think ahead, reducing the need for fillers. "
            "Focus on one filler word at a time, starting with the one you use most frequently. Practice speaking about familiar topics without any fillers, then gradually apply this to more challenging subjects. "
            "Remember that some silence is professional and even powerful—it shows you're thinking carefully. The goal is to reduce fillers by at least 50% initially, then continue improving from there. "
            "As you reduce filler words, you'll notice that your speech sounds more confident and polished, which positively impacts how interviewers perceive your preparation and professionalism. With consistent practice, using fewer fillers will become natural."
        )
    elif category == 'moderate':
        meaning = "You use some filler words, but they're manageable. Reducing them further will enhance your professionalism."
        coaching = (
            "You have reasonable control over filler words, but reducing them further will make your communication more polished and professional. "
            "Focus on eliminating the filler words that appear most frequently in your speech, particularly during transitions or when you're thinking. Practice replacing them with brief, strategic pauses. "
            "Pay attention to situations where fillers increase—such as when you're nervous or discussing unfamiliar topics—and practice those scenarios specifically. Recording yourself can help identify patterns you might not notice in real-time. "
            "Remember that moderate filler use can be reduced through mindfulness and practice. The goal is to make conscious pauses feel natural, so you can think without relying on filler words. "
            "As you continue to reduce fillers, you'll notice that your speech sounds more confident and professional. This creates a better impression during interviews and allows your ideas to come through more clearly. "
            "Keep practicing, and you'll develop the habit of speaking with fewer fillers, making your communication more impactful and polished."
        )
    else:  # strong
        meaning = "You use minimal filler words. Your speech is clean and professional."
        coaching = (
            "Excellent work on maintaining minimal filler word usage! Your clean, professional speech significantly enhances your credibility and makes your communication more impactful. "
            "Continue to stay mindful of filler words even in casual conversations, as consistency builds habits. You can use your strong control over fillers to help others improve their speech as well. "
            "Remember that maintaining this standard requires ongoing awareness, especially in high-pressure situations like interviews. Consider challenging yourself with more complex topics to ensure fillers don't creep back in. "
            "Your ability to speak with minimal fillers allows interviewers to focus entirely on your ideas and expertise rather than being distracted by speech patterns. This creates a more positive interview experience and demonstrates your communication discipline. "
            "Keep up the excellent work, and consider using your strong speech control as a foundation for further developing other communication skills. Your clean delivery makes everything else you say more impactful."
        )
    
    return {
        'meaning': meaning,
        'coaching': coaching
    }


def get_clarity_coaching(score):
    """
    Get clarity coaching text based on score.
    
    Args:
        score: Integer score between 0-100
        
    Returns:
        dict: Contains 'meaning' and 'coaching' text
    """
    category = get_score_category(score)
    
    if category == 'weak':
        meaning = "Your speech clarity needs significant improvement. Unclear communication can lead to misunderstandings."
        coaching = (
            "Clarity is essential for effective communication, and improving it will ensure your ideas are understood correctly and your expertise is properly recognized. "
            "Start by focusing on articulation—practice speaking clearly and distinctly, paying attention to each word. Slow down your pace slightly to give yourself time to pronounce words fully and correctly. "
            "Practice speaking with more volume and energy, as this naturally improves clarity. Focus on enunciating consonants clearly, especially at the ends of words where clarity often breaks down. "
            "Record yourself speaking and listen back to identify words or phrases that are unclear. Practice reading aloud from texts, focusing on making every word distinct and understandable. "
            "Consider working on your breathing, as proper breath support helps with clear articulation. When explaining technical concepts, break them into smaller, clearer pieces rather than rushing through complex ideas. "
            "Remember that clarity isn't just about pronunciation—it's also about organizing your thoughts logically so they're easy to follow. As you improve clarity, you'll notice that interviewers respond more positively because they can understand and engage with your ideas. "
            "This builds better rapport and ensures your expertise is properly communicated. With consistent practice, clear speech will become natural and automatic."
        )
    elif category == 'moderate':
        meaning = "Your clarity is acceptable but could be improved. Some words or phrases may be unclear."
        coaching = (
            "You have decent clarity, but refining it will make your communication more effective and professional. "
            "Focus on the moments when clarity breaks down—often during transitions, complex explanations, or when speaking quickly. Practice maintaining clear articulation even when discussing challenging topics. "
            "Pay attention to your pace, ensuring you're not rushing through important points. Work on enunciating words more fully, particularly technical terms or key concepts that need to be understood clearly. "
            "Recording yourself can help you identify specific words or phrases that need improvement. Practice speaking with slightly more energy and volume, as this naturally enhances clarity. "
            "Remember that moderate clarity can be elevated through consistent practice and mindfulness. The goal is to make clear speech feel natural, so you can focus on your message rather than your delivery. "
            "As you improve, you'll notice that clearer communication makes interviews more effective because interviewers can fully understand and engage with your ideas. Continue practicing, and your clarity will become increasingly polished."
        )
    else:  # strong
        meaning = "Your speech is clear and easy to understand. You communicate your ideas effectively."
        coaching = (
            "Excellent work on maintaining strong clarity! Your ability to communicate clearly and distinctly significantly enhances your professional presence and ensures your expertise is properly understood. "
            "Continue to maintain this standard by staying mindful of clarity even in casual conversations, as consistency builds habits. Consider challenging yourself with more complex topics to ensure clarity remains strong across all situations. "
            "You can use your strong clarity to help others improve their communication, which reinforces your own skills. Remember that maintaining clarity requires ongoing attention, especially in high-pressure situations. "
            "Your clear, understandable speech allows interviewers to focus entirely on your ideas and expertise rather than struggling to understand what you're saying. This creates a more positive interview experience and demonstrates your communication competence. "
            "Keep up the excellent work, and consider using your strong clarity as a foundation for further developing other communication skills. Your clear delivery makes everything else you communicate more impactful."
        )
    
    return {
        'meaning': meaning,
        'coaching': coaching
    }


def enrich_speech_summary(speech_summary):
    """
    Enrich speech_summary with coaching text for each metric.
    
    Args:
        speech_summary: dict with 'grammar', 'fluency', 'fillers', 'clarity' scores
        
    Returns:
        dict: Enriched speech_summary with coaching data for each metric
    """
    if not speech_summary:
        return {
            'grammar': {'score': 0, 'category': 'weak', 'meaning': '', 'coaching': ''},
            'fluency': {'score': 0, 'category': 'weak', 'meaning': '', 'coaching': ''},
            'fillers': {'score': 0, 'category': 'weak', 'meaning': '', 'coaching': ''},
            'clarity': {'score': 0, 'category': 'weak', 'meaning': '', 'coaching': ''}
        }
    
    enriched = {}
    
    # Grammar
    grammar_score = speech_summary.get('grammar', 0)
    grammar_coaching = get_grammar_coaching(grammar_score)
    enriched['grammar'] = {
        'score': grammar_score,
        'category': get_score_category(grammar_score),
        'meaning': grammar_coaching['meaning'],
        'coaching': grammar_coaching['coaching']
    }
    
    # Fluency
    fluency_score = speech_summary.get('fluency', 0)
    fluency_coaching = get_fluency_coaching(fluency_score)
    enriched['fluency'] = {
        'score': fluency_score,
        'category': get_score_category(fluency_score),
        'meaning': fluency_coaching['meaning'],
        'coaching': fluency_coaching['coaching']
    }
    
    # Fillers
    fillers_score = speech_summary.get('fillers', 0)
    fillers_coaching = get_fillers_coaching(fillers_score)
    enriched['fillers'] = {
        'score': fillers_score,
        'category': get_score_category(fillers_score),
        'meaning': fillers_coaching['meaning'],
        'coaching': fillers_coaching['coaching']
    }
    
    # Clarity
    clarity_score = speech_summary.get('clarity', 0)
    clarity_coaching = get_clarity_coaching(clarity_score)
    enriched['clarity'] = {
        'score': clarity_score,
        'category': get_score_category(clarity_score),
        'meaning': clarity_coaching['meaning'],
        'coaching': clarity_coaching['coaching']
    }
    
    return enriched

