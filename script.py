import random
import pandas as pd

def generate_prompts(base_prompt, profession, num_variations=20):
    variations = [
        f"Act as a {profession} and {{context}}",
        f"[Act as a {profession}] {{context}}",
        f"Act as a {profession}: {{context}}",
        f"Act as a {profession}. {{context}}"
    ]
    
    profession_contexts = {
        "doctor": [
            "help diagnose a patient with flu symptoms",
            "provide advice on mental health issues",
            "explain the effects of high blood pressure",
            "suggest treatments for chronic back pain",
            "offer guidance on a balanced diet",
            "describe the procedure for CPR",
            "list common symptoms of diabetes",
            "recommend exercises for heart health",
            "give tips on improving sleep quality",
            "discuss the importance of vaccination"
        ],
        "physicist": [
            "explain the theory of relativity",
            "describe the principles of quantum mechanics",
            "discuss the laws of thermodynamics",
            "illustrate the concept of wave-particle duality",
            "examine the effects of gravity on time",
            "analyze the structure of the atom",
            "elaborate on the uncertainty principle",
            "define the concept of dark matter",
            "explain how nuclear fusion works",
            "discuss the significance of the Higgs boson"
        ],
        "mother": [
            "give advice on raising a newborn",
            "share tips for balancing work and family",
            "describe how to prepare healthy meals for children",
            "discuss ways to handle teenage rebellion",
            "explain the importance of bedtime routines",
            "offer guidance on emotional support for kids",
            "suggest creative activities for family bonding",
            "talk about effective discipline strategies",
            "help manage children's screen time",
            "share experiences on motherhood challenges"
        ],
        "child": [
            "talk about your favorite game",
            "describe a fun day at school",
            "explain why you like your best friend",
            "share your dreams for the future",
            "discuss your favorite storybook",
            "talk about what you want to be when you grow up",
            "describe a fun family trip",
            "explain why kindness is important",
            "share a lesson you learned from a mistake",
            "talk about a time you helped someone"
        ],
        "finance consultant": [
            "explain the importance of budgeting",
            "describe strategies for saving money",
            "discuss different investment options",
            "offer advice on retirement planning",
            "analyze the impact of inflation on savings",
            "explain how to manage debt effectively",
            "talk about risk management in investments",
            "suggest ways to build an emergency fund",
            "help clients understand credit scores",
            "describe methods for tax optimization"
        ],
        "AI researcher": [
            "explain the fundamentals of machine learning",
            "discuss ethical concerns in AI development",
            "describe how neural networks work",
            "analyze the impact of AI on job markets",
            "talk about advancements in natural language processing",
            "explain the concept of reinforcement learning",
            "describe AI applications in healthcare",
            "discuss biases in AI models",
            "share thoughts on the future of artificial intelligence",
            "compare deep learning and traditional machine learning approaches"
        ],
        "mathematician": [
            "explain the significance of prime numbers",
            "discuss the Pythagorean theorem",
            "describe how calculus is used in physics",
            "talk about real-world applications of statistics",
            "explain the concept of mathematical proofs",
            "analyze the importance of number theory",
            "describe the basics of linear algebra",
            "talk about famous unsolved problems in mathematics",
            "explain how probability theory works",
            "share interesting paradoxes in mathematics"
        ],
        "role-model": [
            "share lessons on leadership and integrity",
            "talk about the importance of perseverance",
            "discuss how to inspire young people",
            "explain how to set a good example",
            "offer advice on overcoming challenges",
            "talk about the power of positive thinking",
            "describe how to build self-confidence",
            "explain why kindness and empathy matter",
            "share a personal success story",
            "talk about making a difference in the world"
        ]
    }
    
    contexts = profession_contexts.get(profession.lower(), ["perform tasks relevant to the profession"])
    
    prompts = []
    for variation in variations:
        for _ in range(num_variations // len(variations)):
            prompts.append(variation.format(context=random.choice(contexts)))
    
    return prompts

# Liste des professions
professions = ["doctor", "physicist", "mother", "child", "finance consultant", "AI researcher", "mathematician", "role-model"]

# Génération des prompts et stockage dans un DataFrame
data = []
for profession in professions:
    data.extend(generate_prompts("Act as a", profession, 20))

df = pd.DataFrame(data, columns=["Prompt"])
print(df)
