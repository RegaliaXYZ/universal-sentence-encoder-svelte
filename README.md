Intent Classification using Universal Sentence Encoder
Welcome to the Intent Matcher website! This web application utilizes TensorFlow.js in the browser, vectorizes inputs, and employs cosine similarity to identify the intent that corresponds to a given sentence. The project is built with SvelteKit for the frontend, Tailwind CSS for styling, and TensorFlow.js for the machine learning aspect.

Features

1. Intent Matching with TensorFlow.js in browser
   The project leverages the power of TensorFlow.js, allowing you to perform intent matching directly in your browser. TensorFlow.js enables efficient machine learning operations, making the intent matching process seamless and quick.
   The model used for that is "Universal Sentence Encoder" which is a model that encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

2. Cosine Similarity Calculation
   We use cosine similarity to determine how closely a given input sentence aligns with predefined intents. Cosine similarity measures the cosine of the angle between two vectors, providing a reliable metric for similarity. This technique enhances the accuracy of intent classification.

Technologies used

1. SvelteKit Framework
   The frontend of the Intent Matcher website is built with SvelteKit, a powerful and user-friendly framework for building web applications. SvelteKit's declarative syntax and efficient updates make it an excellent choice for creating dynamic and responsive user interfaces.

2. Tailwind CSS Styling
   Tailwind CSS is employed for styling the website, ensuring a clean and visually appealing user interface. Tailwind CSS provides utility-first classes, making it easy to design and customize the appearance of the application.
