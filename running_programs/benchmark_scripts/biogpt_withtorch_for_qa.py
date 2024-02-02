def run(q2, c2):
	from transformers import AutoModelForCausalLM, AutoTokenizer
	import torch

	#load model and tokenizer
	model = AutoModelForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sacstudent/svanderwal2/")
	tokenizer = AutoTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sascstudent/svanderwal2")

	#tokenize
	#question = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
	#context = "Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.", "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells."
	#input_text = question + " \\n " + context[0]

	#q2 = "Landolt C and snellen e acuity: differences in strabismus amblyopia?"
	#c2 = [ "Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize different optotypes differs even if their critical details appear under the same visual angle. Since optotypes are evaluated on individuals with good visual acuity and without eye disorders, differences in the lower visual acuity range cannot be excluded. In this study, visual acuity measured with the Snellen E was compared to the Landolt C acuity.", "100 patients (age 8 - 90 years, median 60.5 years) with various eye disorders, among them 39 with amblyopia due to strabismus, and 13 healthy volunteers were tested. Charts with the Snellen E and the Landolt C (Precision Vision) which mimic the ETDRS charts were used to assess visual acuity. Three out of 5 optotypes per line had to be correctly identified, while wrong answers were monitored. In the group of patients, the eyes with the lower visual acuity, and the right eyes of the healthy subjects, were evaluated.", "Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small. The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia. The mean difference between LR and SE was 0.55 lines in the entire group and 0.55 lines for the eyes with strabismus amblyopia, with higher values of SE in both groups. The results of the other groups were similar with only small differences between LR and SE." ] 
	input_text = q2 + " \\n " + c2

	inputs = tokenizer.encode(input_text, return_tensors='pt')
	outputs = model.generate(inputs, max_length=512)

	answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
	return answer

