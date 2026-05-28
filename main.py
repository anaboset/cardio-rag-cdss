import asyncio
import sys
from src.generation.pipeline import rag_response
from src.utils.logger import logger

async def interactive_cli():
    print("\n" + "🩺 " + "="*45 + " 🩺")
    print("      CardioCDSS: Clinical Decision Support")
    print("="*50)
    print("Welcome, Doctor. Type 'exit' at any time to quit.")

    while True:
        print("\n" + "-"*50)
        
        # 1. Get Patient Summary
        patient = input("👤 Enter Patient Summary (Age, BP, Comorbidities):\n> ").strip()
        if patient.lower() == 'exit':
            break
        if not patient:
            print("⚠️ Patient summary cannot be empty.")
            continue

        # 2. Get Clinical Query
        query = input("\n🔍 Enter your Clinical Question:\n> ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            print("⚠️ Clinical query cannot be empty.")
            continue

        print("\n🚀 Processing based on authoritative guidelines...")
        
        try:
            # Execute RAG Pipeline (Async)
            response, docs, variants = await rag_response(query, patient)
            
            print("\n🤖 RECOMMENDED MANAGEMENT:")
            print("="*30)
            print(response)
            print("="*30)
            
            print(f"\n📑 Sources analyzed: {len(docs)}")
            print(f"🔄 Search variations used: {len(variants)}")
            
        except Exception as e:
            logger.error(f"System Error: {e}", exc_info=True)
            print(f"❌ An error occurred during processing. Please check the logs.")

    print("\n👋 System shut down. Stay safe, Doctor.")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_cli())
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted. Closing...")
        sys.exit(0)