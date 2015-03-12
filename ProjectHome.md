This project implements MultiStage Cascading for Weka framework. This algorithm was described in:

C. Kaynak and E. Alpaydin, “MultiStage Cascading of Multiple Classifiers: One Man’s Noise is Another Man's Data,” IN PROCEEDINGS OF THE 17TH INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML-2000, pp. 455 - 462, 2000.

MultiStage Cascading algorithm uses sequence of classifiers to produce a class of an instance. The classifier in sequence is used only if previous classifier's confidence is less than a predefined threshold.