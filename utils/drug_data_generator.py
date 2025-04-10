"""
Drug Data Generator for Pharmacy Data Science Platform.

This module creates a comprehensive drug database with detailed information
for 1000+ pharmaceutical products by integrating data from multiple sources.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import logging
from config import DATA_DIR, logger
from tqdm import tqdm
import csv

# Constants
RXNORM_API_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
DAILYMED_API_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
DRUGBANK_OPEN_URL = "https://go.drugbank.com/releases/latest"
OUTPUT_PATH = os.path.join(DATA_DIR, "comprehensive_drugs.csv")

# Drug categories with subcategories
DRUG_CATEGORIES = {
    "Cardiovascular": [
        "Antihypertensive", "Antianginal", "Antiarrhythmic", "Cardiac Glycoside", 
        "Anticoagulant", "Antihyperlipidemic", "Vasodilator", "Beta Blocker",
        "ACE Inhibitor", "Calcium Channel Blocker", "Diuretic"
    ],
    "Respiratory": [
        "Bronchodilator", "Decongestant", "Antihistamine", "Expectorant", 
        "Antitussive", "Mucolytic", "Leukotriene Modifier"
    ],
    "Gastrointestinal": [
        "Antacid", "Antiemetic", "Antidiarrheal", "Laxative", "Proton Pump Inhibitor",
        "H2 Antagonist", "Antispasmodic", "Prokinetic"
    ],
    "Central Nervous System": [
        "Analgesic", "Antipyretic", "Antiepileptic", "Anxiolytic", "Antidepressant",
        "Antipsychotic", "Sedative", "Hypnotic", "Antimigraine", "NSAID", "Opioid",
        "SNRI", "SSRI", "Benzodiazepine"
    ],
    "Antibiotic": [
        "Penicillin", "Cephalosporin", "Aminoglycoside", "Macrolide", "Tetracycline",
        "Quinolone", "Sulfonamide", "Carbapenem", "Glycopeptide"
    ],
    "Antidiabetic": [
        "Insulin", "Sulfonylurea", "Biguanide", "Alpha-Glucosidase Inhibitor", 
        "DPP-4 Inhibitor", "SGLT2 Inhibitor", "GLP-1 Receptor Agonist", "Meglitinide"
    ],
    "Hormonal": [
        "Thyroid Hormone", "Corticosteroid", "Estrogen", "Progestin", "Androgen",
        "Oxytocic", "Growth Hormone", "Antithyroid"
    ],
    "Immunological": [
        "Vaccine", "Immunosuppressant", "Immunostimulant", "Antirheumatic",
        "Monoclonal Antibody", "Interferons"
    ],
    "Dermatological": [
        "Antipruritic", "Antipsoriatic", "Antifungal", "Emollient", "Keratolytic",
        "Antiseptic", "Antibiotic", "Corticosteroid"
    ],
    "Anticancer": [
        "Alkylating Agent", "Antimetabolite", "Plant Alkaloid", "Topoisomerase Inhibitor",
        "Cytotoxic Antibiotic", "Targeted Therapy", "Hormone Therapy", "Immunotherapy"
    ],
    "Antiviral": [
        "Nucleoside Analog", "Protease Inhibitor", "Integrase Inhibitor",
        "Fusion Inhibitor", "Entry Inhibitor", "CCR5 Antagonist"
    ],
    "Antiparasitic": [
        "Antimalarial", "Antihelminthic", "Antiprotozoal"
    ],
    "Musculoskeletal": [
        "Muscle Relaxant", "Antigout", "Antirheumatic", "Bisphosphonate"
    ],
    "Nutritional": [
        "Vitamin", "Mineral", "Amino Acid", "Electrolyte", "Caloric Agent"
    ],
    "Ophthalmological": [
        "Antiglaucoma", "Mydriatic", "Miotic", "Lubricant"
    ],
    "Otological": [
        "Cerumenolytic", "Anti-Infective", "Anti-Inflammatory"
    ]
}

# Major pharmaceutical manufacturers
MANUFACTURERS = [
    "Pfizer", "Roche", "Novartis", "Merck", "Johnson & Johnson", "Sanofi",
    "GlaxoSmithKline", "AbbVie", "Gilead Sciences", "Amgen", "AstraZeneca",
    "Bristol-Myers Squibb", "Eli Lilly", "Boehringer Ingelheim", "Bayer",
    "Teva Pharmaceutical", "Novo Nordisk", "Takeda", "Mylan", "Allergan",
    "Biogen", "Baxter International", "Celgene", "Merck KGaA", "Abbott Laboratories",
    "Genentech", "Regeneron", "Vertex", "Cipla", "Sun Pharmaceutical", "Dr. Reddy's",
    "Lupin", "Aurobindo Pharma", "Zydus Cadila", "Torrent Pharmaceuticals",
    "Intas Pharmaceuticals", "Mankind Pharma", "Alkem Laboratories", "Glenmark",
    "Biocon", "Serum Institute", "Piramal Healthcare", "Strides Pharma", "Alembic Pharmaceuticals"
]

# Common dosage forms
DOSAGE_FORMS = [
    "Tablet", "Capsule", "Extended-Release Tablet", "Extended-Release Capsule",
    "Oral Solution", "Oral Suspension", "Injection", "Intravenous Solution",
    "Topical Cream", "Topical Ointment", "Topical Gel", "Transdermal Patch",
    "Nasal Spray", "Ophthalmic Solution", "Ophthalmic Ointment", "Otic Solution",
    "Rectal Suppository", "Inhalation Solution", "Inhalation Powder",
    "Sublingual Tablet", "Buccal Tablet", "Syrup", "Powder for Reconstitution",
    "Chewable Tablet", "Orally Disintegrating Tablet", "Dispersible Tablet",
    "Effervescent Tablet", "Lozenges", "Drops", "Film-Coated Tablet",
    "Enteric-Coated Tablet", "Vaginal Cream", "Vaginal Tablet", "Vaginal Suppository"
]

# Common gene interactions
GENE_INTERACTIONS = [
    "CYP2D6", "CYP3A4", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2B6", 
    "UGT1A1", "VKORC1", "TPMT", "DPYD", "HLA-B*1502", "HLA-B*5701",
    "SLCO1B1", "G6PD", "IFNL3", "CFTR", "BCHE", "NAT2",
    "COMT", "MTHFR", "ABCB1", "ABCG2", "OPRM1", "CYP2E1"
]

def generate_drug_name(base_name, manufacturer, salt=None):
    """Generate a realistic drug name based on manufacturer and optional salt."""
    suffixes = ["", "XR", "SR", "IR", "CR", "ER", "HCT", "HBr", "HCl", "D", "Plus", "Forte", "IV", "Oral"]
    prefix_prob = 0.3  # 30% chance for manufacturer prefix
    
    if np.random.random() < prefix_prob:
        if len(manufacturer.split()) > 1:
            # Use initials for multi-word manufacturers
            prefix = ''.join(word[0] for word in manufacturer.split())
            name = f"{prefix}-{base_name}"
        else:
            # Use shortened manufacturer name
            prefix = manufacturer[:3]
            name = f"{prefix}{base_name}"
    else:
        name = base_name
    
    # Add salt if provided
    if salt:
        name = f"{name} {salt}"
    
    # Add suffix with 40% probability
    if np.random.random() < 0.4:
        suffix = np.random.choice(suffixes)
        if suffix:
            name = f"{name} {suffix}"
    
    return name

def generate_description(category, subcategory, generic_name):
    """Generate a detailed drug description based on category and subcategory."""
    # Templates for different drug categories
    category_templates = {
        "Cardiovascular": [
            f"{generic_name} is indicated for the treatment of hypertension, either alone or in combination with other antihypertensive agents.",
            f"{generic_name} is a {subcategory.lower()} used in the management of heart failure, hypertension, and related cardiovascular conditions.",
            f"{generic_name} belongs to the class of {subcategory.lower()} medications and is used to treat various cardiovascular disorders including coronary artery disease.",
            f"{generic_name} is primarily used to reduce the risk of cardiovascular events in patients with established cardiovascular disease or multiple risk factors."
        ],
        "Respiratory": [
            f"{generic_name} is a {subcategory.lower()} indicated for the treatment of asthma and chronic obstructive pulmonary disease (COPD).",
            f"{generic_name} helps reduce inflammation and bronchospasm in the airways, improving breathing in patients with respiratory conditions.",
            f"{generic_name} is used for symptomatic relief of respiratory tract disorders characterized by congestion and inflammation.",
            f"{generic_name} is indicated for maintenance treatment of bronchospasm associated with respiratory disorders."
        ],
        "Gastrointestinal": [
            f"{generic_name} is a {subcategory.lower()} used in the treatment of acid-related gastrointestinal disorders.",
            f"{generic_name} provides relief from symptoms associated with gastroesophageal reflux disease (GERD) and peptic ulcer disease.",
            f"{generic_name} is indicated for the treatment of various gastrointestinal motility disorders and functional gastrointestinal syndromes.",
            f"{generic_name} helps manage symptoms and promote healing in patients with inflammatory bowel conditions."
        ],
        "Central Nervous System": [
            f"{generic_name} is a {subcategory.lower()} that acts on the central nervous system to treat conditions such as depression, anxiety, or seizures.",
            f"{generic_name} helps restore the balance of certain natural substances in the brain, improving mood, thoughts, and behavior.",
            f"{generic_name} is indicated for the management of chronic pain conditions through its action on central pain pathways.",
            f"{generic_name} is used in the treatment of neurological and psychiatric disorders by modulating neurotransmitter activity."
        ],
        "Antibiotic": [
            f"{generic_name} is a {subcategory.lower()} antibiotic that inhibits bacterial cell wall synthesis, effective against both gram-positive and gram-negative organisms.",
            f"{generic_name} is indicated for the treatment of infections caused by susceptible strains of designated microorganisms.",
            f"{generic_name} demonstrates bactericidal activity against a wide spectrum of pathogens commonly associated with respiratory, skin, and urinary tract infections.",
            f"{generic_name} is a broad-spectrum antibiotic used to treat various bacterial infections throughout the body."
        ],
        "Antidiabetic": [
            f"{generic_name} is a {subcategory.lower()} medication that helps control blood sugar levels in patients with type 2 diabetes mellitus.",
            f"{generic_name} improves insulin sensitivity and reduces hepatic glucose production, leading to better glycemic control.",
            f"{generic_name} stimulates insulin secretion from pancreatic beta cells, helping to lower blood glucose levels in diabetic patients.",
            f"{generic_name} is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes."
        ]
    }
    
    # Default template for categories not in the template dictionary
    default_templates = [
        f"{generic_name} is a {subcategory.lower()} medication classified under {category.lower()} drugs.",
        f"{generic_name} is used in the treatment of conditions requiring {category.lower()} therapy.",
        f"{generic_name} belongs to the class of {subcategory.lower()} agents within the {category.lower()} category.",
        f"{generic_name} is indicated for therapeutic applications related to {category.lower()} conditions."
    ]
    
    # Select templates based on category
    templates = category_templates.get(category, default_templates)
    
    # Generate primary description
    primary_desc = np.random.choice(templates)
    
    # Additional information sections
    mechanism = f"Mechanism of Action: {generic_name} works by {generate_mechanism_of_action(category, subcategory)}."
    
    pharmacokinetics = generate_pharmacokinetics()
    
    contraindications = "Contraindications: " + generate_contraindications(category)
    
    adverse_effects = "Common Adverse Effects: " + generate_adverse_effects(category)
    
    # Combine all sections
    full_description = f"{primary_desc}\n\n{mechanism}\n\n{pharmacokinetics}\n\n{contraindications}\n\n{adverse_effects}"
    
    return full_description

def generate_mechanism_of_action(category, subcategory):
    """Generate a plausible mechanism of action based on drug category."""
    mechanisms = {
        "Cardiovascular": [
            "inhibiting angiotensin-converting enzyme (ACE), which reduces the production of angiotensin II",
            "blocking calcium channels in vascular smooth muscle and cardiac muscle",
            "selectively blocking beta-adrenergic receptors, reducing the effects of epinephrine",
            "reducing the activity of the renin-angiotensin-aldosterone system",
            "promoting sodium and water excretion by inhibiting sodium reabsorption in the distal tubule"
        ],
        "Respiratory": [
            "antagonizing histamine H1 receptors, reducing allergic responses",
            "stimulating beta-2 adrenergic receptors in bronchial smooth muscle, causing bronchodilation",
            "inhibiting phosphodiesterase enzymes, resulting in bronchodilation and anti-inflammatory effects",
            "blocking leukotriene receptors, reducing bronchial inflammation and constriction",
            "reducing mucus production and improving mucociliary clearance"
        ],
        "Gastrointestinal": [
            "inhibiting the gastric H+/K+ ATPase enzyme system, suppressing gastric acid production",
            "antagonizing histamine H2 receptors in gastric parietal cells, reducing acid secretion",
            "forming a protective barrier over inflamed areas in the gastrointestinal tract",
            "blocking dopamine D2 receptors in the chemoreceptor trigger zone, providing antiemetic effects",
            "stimulating gastrointestinal motility through acetylcholine-like effects"
        ],
        "Central Nervous System": [
            "inhibiting the reuptake of serotonin and norepinephrine in the CNS",
            "enhancing GABAergic neurotransmission, producing anxiolytic and sedative effects",
            "blocking NMDA receptors and modulating glutamate neurotransmission",
            "inhibiting monoamine oxidase, increasing levels of monoamine neurotransmitters",
            "binding to opioid receptors, modulating pain perception pathways"
        ],
        "Antibiotic": [
            "inhibiting bacterial cell wall synthesis by binding to penicillin-binding proteins",
            "binding to the 30S ribosomal subunit, inhibiting bacterial protein synthesis",
            "inhibiting DNA gyrase and topoisomerase IV, preventing bacterial DNA replication",
            "disrupting bacterial cell membrane integrity, leading to cell death",
            "interfering with bacterial folate synthesis, inhibiting DNA and RNA production"
        ],
        "Antidiabetic": [
            "stimulating insulin release from pancreatic beta cells",
            "increasing tissue sensitivity to insulin and reducing hepatic glucose production",
            "inhibiting intestinal alpha-glucosidase enzymes, delaying carbohydrate absorption",
            "inhibiting sodium-glucose cotransporter 2 (SGLT2) in the kidneys, increasing urinary glucose excretion",
            "enhancing incretin effect by inhibiting dipeptidyl peptidase-4 (DPP-4)"
        ]
    }
    
    # Default mechanisms for categories not in the dictionary
    default_mechanisms = [
        "binding to specific receptors in target tissues",
        "modulating biochemical pathways involved in the disease process",
        "inhibiting key enzymes in pathological processes",
        "enhancing physiological defense mechanisms",
        "counteracting pathological changes in tissue function"
    ]
    
    category_mechanisms = mechanisms.get(category, default_mechanisms)
    return np.random.choice(category_mechanisms)

def generate_pharmacokinetics():
    """Generate pharmacokinetic information for a drug."""
    absorption = [
        "Rapidly absorbed from the gastrointestinal tract",
        "Slowly absorbed after oral administration",
        "Absorption is enhanced when taken with food",
        "Absorption is reduced when taken with food",
        "Bioavailability is approximately 60-80% after oral administration",
        "Subject to significant first-pass metabolism resulting in reduced bioavailability",
        "Well absorbed through transdermal application",
        "Absorption is pH-dependent, optimized in acidic environment"
    ]
    
    distribution = [
        "Widely distributed throughout body tissues",
        "Distribution limited primarily to the vascular compartment",
        "Highly protein-bound (approximately 90-95%)",
        "Moderately protein-bound (approximately 60-80%)",
        "Low protein binding (<30%)",
        "Crosses the blood-brain barrier",
        "Does not significantly penetrate the blood-brain barrier",
        "Accumulates in adipose tissue due to lipophilic properties"
    ]
    
    metabolism = [
        "Extensively metabolized in the liver via CYP450 enzymes, primarily CYP3A4",
        "Primarily metabolized by CYP2D6 with genetic polymorphisms affecting rate of metabolism",
        "Undergoes conjugation reactions in the liver",
        "Subject to minimal metabolism, with most of the drug excreted unchanged",
        "Metabolized to active metabolites that contribute to the therapeutic effect",
        "Metabolized via non-CYP450 pathways",
        "Metabolism shows significant inter-individual variability",
        "Metabolized through multiple pathways with no single dominant route"
    ]
    
    elimination = [
        "Eliminated primarily via renal excretion of unchanged drug",
        "Excreted in urine as metabolites",
        "Eliminated through both renal and biliary routes",
        "Terminal half-life of approximately 12-24 hours",
        "Short half-life of 2-4 hours requiring multiple daily dosing",
        "Prolonged half-life allowing once-daily dosing",
        "Clearance is reduced in patients with renal impairment",
        "Elimination rate depends on hepatic function"
    ]
    
    pharmacokinetics = "Pharmacokinetics: " + np.random.choice(absorption) + ". " + np.random.choice(distribution) + ". " + np.random.choice(metabolism) + ". " + np.random.choice(elimination) + "."
    
    return pharmacokinetics

def generate_contraindications(category):
    """Generate contraindications based on drug category."""
    common_contraindications = [
        "Hypersensitivity to the active substance or any of the excipients",
        "Severe hepatic impairment",
        "Severe renal impairment",
        "Pregnancy and lactation"
    ]
    
    specific_contraindications = {
        "Cardiovascular": [
            "Cardiogenic shock",
            "Acute heart failure",
            "Severe bradycardia",
            "Second or third-degree heart block",
            "Bilateral renal artery stenosis"
        ],
        "Respiratory": [
            "Status asthmaticus",
            "Severe COPD with respiratory insufficiency",
            "Known hypersensitivity to adrenergic compounds",
            "Untreated fungal infections of the respiratory tract"
        ],
        "Gastrointestinal": [
            "Gastrointestinal obstruction",
            "Gastrointestinal perforation",
            "Inflammatory bowel disease in acute phase",
            "Active gastrointestinal bleeding"
        ],
        "Central Nervous System": [
            "Concomitant use of monoamine oxidase inhibitors",
            "Seizure disorders",
            "Angle-closure glaucoma",
            "Recent myocardial infarction",
            "Suicidal ideation"
        ],
        "Antibiotic": [
            "Known bacterial resistance",
            "History of antibiotic-associated colitis",
            "Myasthenia gravis (for certain antibiotic classes)",
            "Concurrent use of contraindicated medications"
        ],
        "Antidiabetic": [
            "Diabetic ketoacidosis",
            "Type 1 diabetes mellitus (for certain antidiabetic classes)",
            "Severe renal impairment (GFR < 30 mL/min)",
            "Acute or chronic metabolic acidosis"
        ]
    }
    
    # Get category-specific contraindications
    category_specific = specific_contraindications.get(category, [])
    
    # Combine common and specific contraindications
    combined_contraindications = common_contraindications + category_specific
    
    # Randomly select 3-5 contraindications
    num_contraindications = np.random.randint(3, 6)
    selected_contraindications = np.random.choice(combined_contraindications, size=min(num_contraindications, len(combined_contraindications)), replace=False)
    
    return "; ".join(selected_contraindications) + "."

def generate_adverse_effects(category):
    """Generate adverse effects based on drug category."""
    common_adverse_effects = [
        "Headache",
        "Nausea",
        "Dizziness",
        "Fatigue",
        "Diarrhea",
        "Vomiting",
        "Abdominal pain",
        "Rash",
        "Pruritus"
    ]
    
    specific_adverse_effects = {
        "Cardiovascular": [
            "Hypotension",
            "Bradycardia",
            "Edema",
            "Palpitations",
            "Syncope",
            "Chest pain",
            "Flushing"
        ],
        "Respiratory": [
            "Cough",
            "Dyspnea",
            "Pharyngitis",
            "Rhinitis",
            "Bronchospasm",
            "Increased sputum production"
        ],
        "Gastrointestinal": [
            "Constipation",
            "Flatulence",
            "Dyspepsia",
            "Dry mouth",
            "Altered taste",
            "Increased appetite"
        ],
        "Central Nervous System": [
            "Somnolence",
            "Insomnia",
            "Anxiety",
            "Depression",
            "Tremor",
            "Paresthesia",
            "Confusion",
            "Memory impairment"
        ],
        "Antibiotic": [
            "Antibiotic-associated diarrhea",
            "Candidiasis",
            "Allergic reactions",
            "Photosensitivity",
            "Tooth discoloration",
            "QT interval prolongation"
        ],
        "Antidiabetic": [
            "Hypoglycemia",
            "Weight gain",
            "Peripheral edema",
            "Lactic acidosis",
            "Urinary tract infections",
            "Genital mycotic infections"
        ]
    }
    
    # Get category-specific adverse effects
    category_specific = specific_adverse_effects.get(category, [])
    
    # Combine common and specific adverse effects
    combined_effects = common_adverse_effects + category_specific
    
    # Randomly select 4-7 adverse effects
    num_effects = np.random.randint(4, 8)
    selected_effects = np.random.choice(combined_effects, size=min(num_effects, len(combined_effects)), replace=False)
    
    return ", ".join(selected_effects) + "."

def generate_dosage_info(drug_name, dosage_form, category):
    """Generate realistic dosage information based on category and form."""
    strength_units = {
        "Tablet": ["mg", "mcg", "g"],
        "Capsule": ["mg", "mcg", "g"],
        "Extended-Release Tablet": ["mg", "mcg"],
        "Extended-Release Capsule": ["mg", "mcg"],
        "Oral Solution": ["mg/mL", "mcg/mL", "%"],
        "Oral Suspension": ["mg/mL", "mcg/mL", "%"],
        "Injection": ["mg/mL", "mcg/mL", "mg", "g", "IU/mL"],
        "Intravenous Solution": ["mg/mL", "mcg/mL", "%", "mEq/mL"],
        "Topical Cream": ["%", "mg/g"],
        "Topical Ointment": ["%", "mg/g"],
        "Topical Gel": ["%", "mg/g"],
        "Transdermal Patch": ["mg/24h", "mcg/h"],
        "Nasal Spray": ["mg/dose", "mcg/dose", "%"],
        "Ophthalmic Solution": ["%", "mg/mL"],
        "Ophthalmic Ointment": ["%", "mg/g"],
        "Otic Solution": ["%", "mg/mL"],
        "Inhalation Solution": ["mg/mL", "mcg/mL", "%"],
        "Inhalation Powder": ["mcg/dose", "mg/dose"],
        "Syrup": ["mg/5mL", "mg/mL", "%"],
    }
    
    # Default to mg if dosage form not found
    default_unit = "mg"
    
    # Get appropriate units for the dosage form
    possible_units = strength_units.get(dosage_form, [default_unit])
    selected_unit = np.random.choice(possible_units)
    
    # Generate strength based on unit type
    if selected_unit == "%":
        strength_values = [0.025, 0.05, 0.1, 0.5, 1, 2, 2.5, 5, 10]
        strength = np.random.choice(strength_values)
    elif selected_unit == "mg/mL":
        strength_values = [0.5, 1, 2, 5, 10, 20, 25, 50, 100]
        strength = np.random.choice(strength_values)
    elif selected_unit == "mcg/mL":
        strength_values = [5, 10, 20, 25, 50, 100, 200, 500, 1000]
        strength = np.random.choice(strength_values)
    elif selected_unit == "mcg":
        strength_values = [5, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
        strength = np.random.choice(strength_values)
    elif selected_unit == "mcg/dose" or selected_unit == "mcg/h":
        strength_values = [5, 10, 25, 50, 100, 200, 400, 500]
        strength = np.random.choice(strength_values)
    elif selected_unit == "g":
        strength_values = [0.5, 1, 2]
        strength = np.random.choice(strength_values)
    elif selected_unit == "IU/mL":
        strength_values = [10, 20, 40, 50, 100, 200, 500, 1000, 10000]
        strength = np.random.choice(strength_values)
    elif selected_unit == "mEq/mL":
        strength_values = [0.5, 1, 2, 4]
        strength = np.random.choice(strength_values)
    else:  # Default to mg
        category_strength_ranges = {
            "Cardiovascular": [2.5, 5, 10, 20, 25, 40, 50, 80, 100, 150, 200, 300],
            "Respiratory": [2, 4, 5, 10, 20, 40, 50, 100, 200, 400],
            "Gastrointestinal": [10, 15, 20, 30, 40, 50, 60, 100, 150, 300],
            "Central Nervous System": [5, 10, 15, 20, 25, 37.5, 50, 75, 100, 150, 200, 300],
            "Antibiotic": [100, 125, 250, 500, 750, 850, 1000],
            "Antidiabetic": [0.5, 1, 2, 2.5, 5, 10, 15, 25, 50, 100],
            "Hormonal": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2],
            "Dermatological": [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25, 40],
        }
        
        strength_values = category_strength_ranges.get(category, [5, 10, 20, 25, 50, 100, 200, 500])
        strength = np.random.choice(strength_values)
    
    # Format strength with unit
    strength_text = f"{strength} {selected_unit}"
    
    # Generate dosage forms (package content)
    if "Tablet" in dosage_form or "Capsule" in dosage_form:
        package_sizes = [10, 14, 20, 28, 30, 50, 60, 90, 100, 120, 180]
        package_content = f"{np.random.choice(package_sizes)} {dosage_form}s per bottle"
    elif "Solution" in dosage_form or "Suspension" in dosage_form or "Syrup" in dosage_form:
        volumes = [10, 15, 30, 60, 100, 120, 150, 240, 480]
        package_content = f"{np.random.choice(volumes)} mL bottle"
    elif "Cream" in dosage_form or "Ointment" in dosage_form or "Gel" in dosage_form:
        sizes = [5, 10, 15, 20, 30, 45, 50, 60, 100]
        package_content = f"{np.random.choice(sizes)} g tube"
    elif "Injection" in dosage_form:
        volumes = [1, 2, 5, 10, 20, 50]
        units = [1, 5, 10, 25, 50]
        package_content = f"{np.random.choice(units)} {np.random.choice(volumes)} mL vials"
    elif "Patch" in dosage_form:
        units = [4, 5, 7, 10, 14, 30]
        package_content = f"{np.random.choice(units)} patches per box"
    elif "Spray" in dosage_form:
        sizes = [10, 15, 20, 30]
        package_content = f"{np.random.choice(sizes)} mL bottle"
    elif "Inhalation" in dosage_form:
        if "Powder" in dosage_form:
            doses = [30, 60, 90, 120, 180, 200]
            package_content = f"{np.random.choice(doses)} doses per inhaler"
        else:
            volumes = [10, 15, 20, 30]
            package_content = f"{np.random.choice(volumes)} mL bottle"
    elif "Ophthalmic" in dosage_form or "Otic" in dosage_form:
        volumes = [2.5, 5, 7.5, 10, 15]
        package_content = f"{np.random.choice(volumes)} mL bottle"
    else:
        units = [10, 20, 30, 50, 100]
        package_content = f"{np.random.choice(units)} units per package"
    
    # Generate frequency based on category
    frequencies = {
        "Cardiovascular": ["once daily", "twice daily", "three times daily"],
        "Antibiotic": ["twice daily", "three times daily", "four times daily"],
        "Antidiabetic": ["once daily", "twice daily", "with meals"],
        "Central Nervous System": ["once daily", "twice daily", "at bedtime", "as needed for pain"],
        "Respiratory": ["twice daily", "every 4-6 hours as needed", "once daily"],
    }
    
    default_frequencies = ["once daily", "twice daily", "three times daily", "as directed by physician"]
    frequency = np.random.choice(frequencies.get(category, default_frequencies))
    
    # Combine information
    dosage_info = f"{strength_text} {dosage_form}, {package_content}. Typical dosing: {frequency}."
    
    return dosage_info

def generate_drug_info(id):
    """Generate comprehensive information for a single drug."""
    # Select random category and subcategory
    category = np.random.choice(list(DRUG_CATEGORIES.keys()))
    subcategory = np.random.choice(DRUG_CATEGORIES[category])
    
    # Generate generic name (base name)
    base_prefixes = ["ab", "ac", "az", "be", "bro", "ca", "ce", "ci", "clo", "de", "di", "do", "en", "es", "fe", "fi", "flu",
                    "ga", "ge", "gli", "he", "hy", "in", "ke", "la", "le", "li", "lo", "ma", "me", "mi", "mo", "na", "ne",
                    "no", "pa", "pe", "pro", "qu", "ra", "re", "ro", "se", "si", "sta", "su", "ta", "te", "ti", "to", "va",
                    "ve", "vi", "zo"]
    
    base_suffixes = ["vir", "zole", "pril", "sartan", "statin", "olol", "dipine", "micin", "ciclovir", "thiazide", "floxacin",
                     "caine", "afil", "antine", "anide", "azine", "fenac", "fibrate", "gliptin", "lizid", "methasone", "mycin",
                     "oxacin", "profen", "ridone", "semide", "setron", "thiazide", "triptan", "vastatin", "zosin", "racil", 
                     "dronate", "phylline", "platin", "toin", "zepam", "zolam", "zepine", "tidine", "tuximab", "parib", "tinib",
                     "cycline", "conazole", "lukast", "navir", "prazole", "glitazone", "mustine", "citabine", "rubicin"]
    
    category_suffixes = {
        "Cardiovascular": ["pril", "sartan", "olol", "dipine", "statin", "fibrate", "semide", "thiazide", "xaban"],
        "Respiratory": ["terol", "tide", "lukast", "lizid", "phylline", "cromil", "tiderol", "triptium"],
        "Gastrointestinal": ["prazole", "tidine", "pride", "setron", "sal", "tate", "caine"],
        "Central Nervous System": ["zepam", "zolam", "zepine", "triptyline", "traline", "pramine", "done", "ridone", "triptan"],
        "Antibiotic": ["cillin", "mycin", "cycline", "floxacin", "oxacin", "zolid", "micin"],
        "Antidiabetic": ["gliptin", "gliflozin", "glitazone", "glinide", "tide", "glutide"],
        "Hormonal": ["steride", "vatib", "strol", "sterone", "relin", "tropin", "gestin"],
        "Anticancer": ["platin", "taxel", "rubicin", "mustine", "citabine", "tinib", "mab", "parib"]
    }
    
    # Base name with appropriate suffix for category
    if category in category_suffixes and np.random.random() < 0.7:
        # 70% chance to use category-specific suffix
        suffix = np.random.choice(category_suffixes[category])
    else:
        suffix = np.random.choice(base_suffixes)
    
    prefix = np.random.choice(base_prefixes)
    base_name = prefix + suffix
    
    # Possible salt forms for some drugs
    salts = [None, "Hydrochloride", "Sodium", "Potassium", "Calcium", "Maleate", "Citrate", "Sulfate", "Phosphate", "Nitrate"]
    salt_weight = [0.7, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02]  # 70% chance of no salt
    salt = np.random.choice(salts, p=salt_weight)
    
    # Manufacturer
    manufacturer = np.random.choice(MANUFACTURERS)
    
    # Generate brand name
    brand_name = generate_drug_name(base_name, manufacturer, salt)
    
    # Dosage form
    dosage_form = np.random.choice(DOSAGE_FORMS)
    
    # Generate NDC (National Drug Code) - format 12345-678-90
    ndc = f"{np.random.randint(10000, 99999)}-{np.random.randint(100, 999)}-{np.random.randint(10, 99)}"
    
    # Generate detailed description
    description = generate_description(category, subcategory, base_name)
    
    # Generate dosage information
    dosage = generate_dosage_info(brand_name, dosage_form, category)
    
    # Randomly assign gene interactions (30% chance of having gene interactions)
    has_gene_interaction = np.random.random() < 0.3
    gene_interactions = []
    if has_gene_interaction:
        num_interactions = np.random.randint(1, 4)
        gene_interactions = np.random.choice(GENE_INTERACTIONS, size=num_interactions, replace=False).tolist()
    
    # Generate price ranges based on category and dosage form
    base_price_ranges = {
        "Cardiovascular": (15, 150),
        "Respiratory": (20, 200),
        "Gastrointestinal": (25, 175),
        "Central Nervous System": (30, 250),
        "Antibiotic": (10, 120),
        "Antidiabetic": (25, 300),
        "Hormonal": (20, 220),
        "Immunological": (50, 500),
        "Dermatological": (15, 150),
        "Anticancer": (200, 2000),
        "Antiviral": (100, 1000),
        "Antiparasitic": (15, 120),
        "Musculoskeletal": (20, 180),
        "Nutritional": (5, 50),
        "Ophthalmological": (25, 150),
        "Otological": (15, 100)
    }
    
    # Get price range for the category
    min_price, max_price = base_price_ranges.get(category, (20, 200))
    
    # Adjust price based on dosage form
    form_multipliers = {
        "Extended-Release": 1.5,
        "Extended-Release Capsule": 1.5,
        "Injection": 2.0,
        "Intravenous Solution": 2.5,
        "Ophthalmic": 1.3,
        "Transdermal Patch": 1.8,
        "Inhalation": 1.7
    }
    
    # Apply multiplier if dosage form contains any of the keys
    for form_key, multiplier in form_multipliers.items():
        if form_key in dosage_form:
            min_price *= multiplier
            max_price *= multiplier
            break
    
    # Generate random price
    price = round(np.random.uniform(min_price, max_price), 2)
    
    # Generate stock level - normal distribution around mean of 500
    stock = max(0, int(np.random.normal(500, 150)))
    
    # Create drug record
    drug = {
        "id": id,
        "generic_name": base_name.capitalize(),
        "brand_name": brand_name,
        "manufacturer": manufacturer,
        "category": category,
        "subcategory": subcategory,
        "ndc": ndc,
        "price": price,
        "dosage_form": dosage_form,
        "dosage": dosage,
        "description": description,
        "gene_interactions": ",".join(gene_interactions) if gene_interactions else "",
        "stock": stock
    }
    
    return drug

def generate_comprehensive_drug_database(num_drugs=1000, output_path=None):
    """Generate a comprehensive drug database with detailed information."""
    if output_path is None:
        output_path = OUTPUT_PATH
    
    logger.info(f"Generating comprehensive drug database with {num_drugs} entries...")
    
    # Create drug records
    drugs = []
    for i in tqdm(range(1, num_drugs + 1)):
        drug = generate_drug_info(i)
        drugs.append(drug)
    
    # Create DataFrame
    drugs_df = pd.DataFrame(drugs)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    drugs_df.to_csv(output_path, index=False)
    logger.info(f"Drug database saved to {output_path}")
    
    return drugs_df

def fetch_rxnorm_data(drug_name):
    """Fetch RxNorm data for a drug (placeholder for actual API call)."""
    try:
        # In real implementation, this would call the RxNorm API
        params = {"name": drug_name, "search": "1"}
        response = requests.get(RXNORM_API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Process response
            return data
        else:
            logger.warning(f"RxNorm API returned status code {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching RxNorm data: {e}")
        return None

def enrich_with_rxnorm_data(drugs_df, limit=100):
    """Enrich drug data with RxNorm information (demonstration)."""
    logger.info("Enriching drug data with RxNorm information...")
    
    # In a real implementation, this would make actual API calls
    # For demonstration, we'll process a limited number of drugs
    sample_size = min(limit, len(drugs_df))
    
    for i in tqdm(range(sample_size)):
        drug_name = drugs_df.iloc[i]['generic_name']
        # This would be replaced with actual API calls in production
        rxnorm_data = fetch_rxnorm_data(drug_name)
        
        # Process the data (placeholder)
        if rxnorm_data:
            # Update drug record with RxNorm information
            pass
    
    return drugs_df

def main():
    """Generate and save comprehensive drug database."""
    try:
        # Generate 1000 detailed drug records
        drugs_df = generate_comprehensive_drug_database(1000)
        
        # Optionally enrich with additional data (demonstration)
        # drugs_df = enrich_with_rxnorm_data(drugs_df)
        
        # Display summary
        print(f"Generated {len(drugs_df)} drug records")
        print(f"Data saved to {OUTPUT_PATH}")
        
        # Display sample
        print("\nSample drug records:")
        print(drugs_df.sample(5)[['brand_name', 'manufacturer', 'category', 'price']].to_string())
        
        return drugs_df
    
    except Exception as e:
        logger.error(f"Error generating drug database: {e}")
        raise

if __name__ == "__main__":
    main()