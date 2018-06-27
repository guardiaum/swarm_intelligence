
### Intruções para execução de experimentos
- Diretivas comuns em todos os algoritmos:
    ```
    --dataset <nome do dataset> : pode ser digits, iris, seeds, heart, cancer e wine
    --alg <nome do algoritmo> : cs, bp, pso, fips, ring, psow
    --hidden <qtd neurônios na camada escondida>
    --maxiter <máximo de interações (ou épocas, no caso do backpropagation)>
    --trial <quantidade de vezes a executar cada algoritmo>
    ```

- Diretivas para todos os algoritmos, exceto o backpropagation:
    ```
    --p <tamanho da população de ovos ou partículas>
    --gloss <valor do gloss>
    ```

- Diretiva específica do backpropagation: 
    ```
    --rate <valor do learning rate>
    ```

- Diretivas específicas de todos os PSO's, exceto PSO-W:
    ```
    --inertia <valor da inércia>
    ```

- Específicos do CLASSIC PSO, do PSO-W e do RING:
    ```
    --c1 <valor>
    --c2 <valor>
    ```

- Específicos do FIPS e RING:
    ```
    --k <quantidade de vizinhos>
    ```

- Exclusivo do FIPS:
    ```
    --wmethod <weight method>: static, distance ou fitness
    ```

##### Exemplos de linhas de comando:

- **Cuckoo**:	
    > _$ python3 Main.py --dataset iris --hidden 3 --trial 30 --maxiter 1000 --gloss 500 --alg cs --p 30_

- **FIPS**: 	
    > _$ python3 Main.py --hidden 3 --trial 30 --maxiter 1000 --gloss 500 --alg fips --p 30 --wmethod static --inertia 0.8 --k 2 --dataset iris_

- **Backpropagation**:
    > _$ python3 Main.py --dataset iris --hidden 3 --trial 30 --maxiter 1000 --gloss 500 --alg bp --rate 0.3_

- **PSO Clássico**:
    > _$ python3 Main.py --dataset iris --hidden 3 --trial 30 --maxiter 1000 --gloss 500 --alg pso --c1 2.55 --c2 2.55 --p 30 --inertia 0.2_

- **PSO_W**
    > _$ python3 Main.py --dataset iris --hidden 3 --trial 30 --maxiter 1000 --gloss 500 --alg psow --c1 2.55 --c2 2.55 --p 30_

- **RING**:
    > _$ python3 Main.py --dataset iris --hidden 3 --trial 30 --maxiter 1000 --gloss 500 --alg ring --c1 2.55 --c2 2.55 --p 30 --inertia 0.2 --k 2_


