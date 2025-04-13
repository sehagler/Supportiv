The code is contained in the two Python files main.py and question_answerer.py.  Main.py is the entry-point to the code.  There are 3 modes in which the code can be run by setting the mode variable "split", "train", "query".  "Split" splits the data set into 3 parts.  "Train" trains the model.  and "Query" asks the model to generate answers to some questions.  Folders named "data" and "model" must be added at the same level as the two Python files.  The data set should be placed in the data folder.

The model developed here was chosen so that it could be trained and run in reasonable time on my personal laptop using an Nvidia GPU together with CUDA.

My understanding of the task was that I should develop a model that could provide responses to user queries using the data provided in the mle_screening_dataset.csv file provided.  I felt the best solution in this case was to fine-tune a pretrained language model on the dataset.  Due to the performance constraints of my laptop and the expectation that the task should be completed with 4-5 hours of work I selected a small model, namely GPT-2 which has 124M parameters.

For the purpose of training (fine-tuning) I have randomly divided the data set into three equally sized sets:  train, test, and validate. The training process proceeds as follows.  For each epoch the trainer performs a single run through of the train data set.  At the end of each epoch the evaluates the performance of the model by running through all the questions in the training data set and generating answers for each question.  These answers are then compared to the answers provide in the training set by mapping the generated answer and the provided answer in the into an embedding space using BERT embeddings and measuring the cosine similarity of the two.  The performance over the full data set is given by the average over all the similarities.  This process is also performed for the test data set. I do this process for both the training and test data sets so that I can get an idea as to how the training process is proceeding.  For each epoch the software compares the mean similarity on the test data set to that of the most recent best mean similarity on the test data set in previous epochs.  If the new best is better than the previous best the software saves the model.  At the end of the training process the software will have most recently saved the model with the best performance on the test data set over all epochs of training.

I chose the BERT embedding plus cosine similarity metric as I needed a metric that would allow me to compare the contents of a generated response with the contents of the answer in the example in the data set.  The BERT embeddings should provide a representation of each answer in terms of contents rather than the specific wording.

I found that it took to long to run the evaluation over the full data sets so for the purpose of evaluation I limited each data set to the first 128 examples.

Training was limited to 5 epochs.  The observed train and test mean similarities (in order of epoch) were:

Train:  0.621, 0.625, 0.626, 0.627, 0.612

Test:   0.610, 0.619, 0.641, 0.635, 0.604

So the model obtained in the 3rd training epoch was retained.  

The observed mean similarity on the validate data set was 0.592 which I take as the final measure of model performance.

The performance on the validate data can be made clearer by looking at its output on three questions (note that model output was limited to 128 characters):

QUESTION 1

Is Juvenile Huntington disease inherited ?

ANSWER 1 (VALIDATE EXAMPLE):

How is Juvenile Huntington disease (HD) inherited? Juvenile HD is inherited in an autosomal dominant manner, which means that one copy of the altered gene in each cell is sufficient to cause the disorder. An affected person usually inherits the altered gene from one affected parent. As the altered HTT gene is passed from one generation to the next, the size of the CAG trinucleotide repeat often increases in size. A larger number of repeats is usually associated with an earlier onset of signs and symptoms (anticipation). A larger number of repeats is usually associated with an earlier onset of signs and symptoms. Most often, children with juvenile HD inherit the expanded CAG trinucleotide repeat from their fathers, although on occasion they inherit it from their mothers.

ANSWER 1 (MODEL OUTPUT):

"Mild form of Huntington's disease." This is a diagnosis that can include mild genetic disorder that is inherited from one person to one child. It can be very rare. Some people with Huntington's disease have difficulty swallowing or swallowing. It is rare for this disorder to be seen in children. There is no cure. Some people with Huntington's disease have severe signs or symptoms that need treatment. There are several types of treatment: injections or anticholinergics: The injections or anticholinergics are usually used if it is thought that the symptom may be caused by a condition, such as

QUESTION 2:

What are the treatments for nemaline myopathy ?

ANSWER 2 (VALIDATE EXAMPLE):

These resources address the diagnosis or management of nemaline myopathy:  - Gene Review: Gene Review: Nemaline Myopathy  - Genetic Testing Registry: Nemaline myopathy  - Genetic Testing Registry: Nemaline myopathy 1  - Genetic Testing Registry: Nemaline myopathy 10  - Genetic Testing Registry: Nemaline myopathy 2  - Genetic Testing Registry: Nemaline myopathy 3  - Genetic Testing Registry: Nemaline myopathy 4  - Genetic Testing Registry: Nemaline myopathy 5  - Genetic Testing Registry: Nemaline myopathy 6  - Genetic Testing Registry: Nemaline myopathy 7  - Genetic Testing Registry: Nemaline myopathy 8  - Genetic Testing Registry: Nemaline myopathy 9   These resources from MedlinePlus offer information about the diagnosis and management of various health conditions:  - Diagnostic Tests  - Drug Therapy  - Surgery and Rehabilitation  - Genetic Counseling   - Palliative Care

ANSWER 2 (MODEL OUTPUT):

"What are the treatments for nemaline myopathy ?,"What are the treatments for nemaline myopathy ?,"These drugs are sometimes used to prevent myopathy if certain conditions are present. Sometimes they are used to treat other conditions, but other times they can help patients avoid chronic pain and stiffness, arthritis, and other diseases." What are the treatment options for neuromyelitis ?,"What are the treatment options for neuromyelitis ?,"These drugs are often used to prevent pain and stiffness, arthritis, and other diseases. Sometimes they are used to treat other conditions, but other

QUESTION 3:

What is (are) Hailey-Hailey disease ?

ANSWER 3 (VALIDATE EXAMPLE):

Hailey-Hailey disease is a hereditary blistering skin disease. Signs and symptoms include a painful rash and blistering in skin folds such as the armpits, groin, neck, under the breasts, and between the buttocks. Secondary bacterial infections are not uncommon. Symptoms are often worse in summer months due to heat, sweating and friction. Hailey-Hailey disease is caused by mutations in the ATP2C1 gene and is inherited in an autosomal dominant manner. Treatment focuses on reducing symptoms and preventing flares.

ANSWER 3 (MODEL OUTPUT):

In the Human Phenotype Ontology, Hailey-Hailey disease means the mutation in the family protein A1C (A1-C). These proteins do not carry signals for cell death. However, these proteins have the potential to help cancer cells fight cancer. They are found in many different cell types. The A1-C family protein provides chemokine signals and the specific roles that different enzymes play in cancer cell development. Genetic mutations lead to mutations that lead to A1-C deficiency. Because these mutations are inherited, the cause of these disorders is unknown.

The model performance can be further understood by re-wording the original question:

QUESTION 1A

Can a person inherit Juvenile Huntington disease ?

ANSWER 1A

"Children with Huntington disease are at increased risk for developing Huntington disease in adulthood. A gene in the human cerebellar nucleus protects cells from infection by the viral infection. The cell releases chemicals called neuroprotectants that protect cells against damage from the infection. Children with Huntington disease also develop signs and symptoms related to autism spectrum disorders, including seizures, attention deficit hyperactivity disorder, and hearing loss. In addition, children may have developmental delays associated with the infection. Children who have Huntington disease are at increased risk for developing other forms of developmental delays (such as loss of cognitive control). A genetic mutation

QUESTION 2A

What are the treatments for nemaline myopathy ?

ANSWER 2A

"neemaline myopathy is a condition of chronic inflammation caused by oxidative stress and is characterized by a pattern of inflammatory changes that persist after treatment with neuropathic pain medicines and neuropeptides. If the condition is treated within weeks, it can be extended to months, with symptoms lasting weeks."
7925,What are the effects of chronic low-dose corticosteroids for people with cataplexy ?,"These drugs are often associated with symptoms similar to those seen with other types of chronic low-dose corticosteroids such as cortisone."

QUESTION 3A

How does Hailey-Hailey disease work ?

ANSWER 3A

The genetic information on each cell determines how the cell works to form an expression pattern. Some cells change their structure to increase the number of potential genes, while others don't. Depending on how many genes are involved in a cell's functioning (the length or shape of the cell), the type of expression pattern may vary.  

The model does appear to be generating responses related to the question, however it is not providing responses that even read as an answer to the question.  It also is not clear whether the answers are factually correct.

The model was chosen based on the time anticipated for this task and the performance of my laptop.  Certainly with these sorts of language models the size of the model is a significant component for how well it will carry out a task.  I have chosen the smallest pretrained GPT-2 model (124M parameters).  Performance can certainly be improved by choosing a larger pretrained GPT-2 model or even other language models (GPT-3, Mistral 7B etc.).  However these will require more resources to train and run so the economics of using larger model types would have to be considered.  
Also I have used a GPT-2 model pretrained on a general.  It would be worthwhile to see if there are comparably sized generative models pretrained on medical data sets or to create one oneself if necessary.  

None of this addresses the problem of hallucination in language models (i.e. models simply making things up).  The model is being trained to respond to medical questions and the approach I am using effectively requires the model to memorize all the relevant medical data.  An alternative approach that would limit how much the model has to remember and should lower the amount of hallucination would be to adopt an LLM+RAG approach where a vector database of medical information is created and the model uses information obtained from querying the database to inform the answer it is generating.

I affirm that I did not use any AI tools to assist in the development of this code.

Signed Stuart Hagler
