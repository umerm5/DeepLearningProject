# DeepLearningProject

Source_mdl.py is the file used to train a source domain model on the images acquired from the ISIS melanoma cancer detection competition,
source_to_test_domain.py is used to predict the test domain images acquired from dermquest from the trained source domian model, and finally domain_adapted_mdl.py is the file used to fine tune the pre-trained model on the source domain by freezing some of its layers
