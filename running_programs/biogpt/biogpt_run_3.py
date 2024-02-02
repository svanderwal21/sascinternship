from transformers import pipeline


model_checkpoint = "/exports/sascstudent/svanderwal2/programs/BioGPT-Large"
qa = pipeline("text-generation", model=model_checkpoint)

context="You are a bio-informatician assistant"
question="How are you doing?"
qa(question=question, context=context)
