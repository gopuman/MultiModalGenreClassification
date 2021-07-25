# MultiModalGenreClassification

IMDb is the biggest online database of data related to movies, TV programmes, video games, and streaming content online- counting cast, plot outlines, appraisals, and open reviews. The task of tagging each movie with its corresponding genre is a manual process and requires users to submit their suggestions, which are then reviewed by an editor.

The aim of the application is to automate this process, by predicting the genres of the movies, based on their posters as well as their summaries. Since one movie is not restricted to a single genre, this is a multi-label problem. 

The project uses two different modalities to solve this problem; one being the image (poster) and the second being the overview of the movie. 

The final outcome of this application is to categorize any given English movie into its corresponding genre(s), based on its overview and poster individually, and then also train a model with the combination of the overview and poster of the movie, to ultimately draw a conclusion as to which model works better. 

