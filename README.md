# Falcon Blei lab LDA interview project    
This project implements LDA on the arvix docs as requested in the email.    

## To run:    
1. Clone the project    
    ```bash
    git clone https://github.com/williamFalcon/falcon-blei-lab-proj.git
    ```    
    
2. Create a virtualenv in the project folder   
    ```bash    
    cd falcon-blei-lab-proj/
    virtualenv .
    ```    

3. Enter the virtualenv    
    ```bash
    source bin/activate
    ```    
    
4. Run the easy_build script    
    ```bash   
    bash easy_build.sh
    ```    

5. If 4 fails:    
    - install and compile lda locally    
    - Copy the lda binary to falcon-blei-lab-proj/app/lda_lib/    
    - pip install -r requirements.txt        

6. From root dir start the python cli server    
    ```bash
    python application.py
    ```    

7. Once the cli server is live, enter a document id to generate the results    
    ```bash
    Enter doc id:  astro-ph/0501545
    -----------------------------
    KL RESULTS
    1. astro-ph/0501545 0811.4158 0.248935443629
    2. astro-ph/0501545 astro-ph/0602198 0.269982883811
    3. astro-ph/0501545 astro-ph/9611163 0.271024548749
    4. astro-ph/0501545 0709.0604 0.34269859565
    5. astro-ph/0501545 0704.1456 0.438767174455
    6. astro-ph/0501545 astro-ph/0105104 0.951945838277
    7. astro-ph/0501545 1604.03369 1.10437234884
    8. astro-ph/0501545 astro-ph/9804085 1.1527657697
    9. astro-ph/0501545 0803.0314 1.37168373674
    10. astro-ph/0501545 astro-ph/0002381 1.50787892437


    -----------------------------
    JS RESULTS
    1. astro-ph/0501545 astro-ph/0602198 0.0292745985366
    2. astro-ph/0501545 astro-ph/9611163 0.035013788712
    3. astro-ph/0501545 0811.4158 0.0646164947416
    4. astro-ph/0501545 0709.0604 0.0960035292377
    5. astro-ph/0501545 0704.1456 0.101831836954
    6. astro-ph/0501545 astro-ph/0105104 0.253682141505
    7. astro-ph/0501545 1604.03369 0.280720993991
    8. astro-ph/0501545 astro-ph/9804085 0.299926858498
    9. astro-ph/0501545 0803.0314 0.355109547897
    10. astro-ph/0501545 astro-ph/0002381 0.376974371766
    ```

## Implementation details    
application.py does the following:    

1. Converts the arvix files to lda-c files.   
2. Runs lda est c-code from python to generate model.    
3. Runs lda inf c-code to inference the test docs.   
4. Precomputes the kl and JS distances for the 100 docs.    
5. starts a cli server where it will wait for a valid document input id.    

## Author    
William Falcon    
waf2107@columbia.edu    
