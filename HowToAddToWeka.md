You can install this implementation of Multistage cascading algorithm in tow ways:
  1. Use Weka 3.7.`*` - it is unstable version, but it has a package manager. With help of it you can install additional packages with classifiers, clustering algorithms etc. If you dare to use unstable version, just download the [package](http://code.google.com/p/multi-stage-cascading-for-weka/downloads/list) and add it to Weka, as described [here](http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F) (see "GUI package manager" section).
  1. To use this classifier with stable version of Weka (3.6.`*`) you need to download Weka sources, add classes from this project to the downloaded sources and build Weka. This will create a .jar file with Weka and MultiStage Cascade classifier. You need just to replace original weka.jar, in the installation directory of Weka, with the one you've built.