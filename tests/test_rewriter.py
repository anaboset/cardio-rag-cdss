import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generation.rewriter import generate_query_variants

def test_rewriter():
    test_queries = [
        "What are the latest guidelines for treating stage 2 hypertension?",
        "Dosage for Atorvastatin in elderly patients with CKD.",
        "Management of acute myocardial infarction."
    ]
    
    print("\n🧪 --- STARTING REWRITER ISOLATION TEST --- 🧪")
    
    for original in test_queries:
        print(f"\n[Original]: {original}")
        variants = generate_query_variants(original)
        
        # Validation checks
        if len(variants) > 1:
            print(f"✅ Success: Generated {len(variants)-1} extra variants.")
            for i, v in enumerate(variants):
                print(f"  {i}. {v}")
        else:
            print("❌ Failure: Only returned the original query.")

    print("\n🧪 --- TEST COMPLETE --- 🧪")

if __name__ == "__main__":
    test_rewriter()