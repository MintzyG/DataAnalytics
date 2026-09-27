# Justificativas do dataset

## Início em 2003

O dataset inclui temporadas a partir de 2003. Esse período oferece dados mais úteis para treinar e avaliar um modelo voltado a corridas recentes.

### Tempos de treino e classificação

Tempos de classificação e treino, incluindo Q1, Q2 e Q3, ficam mais disponíveis e consistentes nesse período. Nas temporadas anteriores, essas colunas têm muitos valores ausentes.

### Inscrições e formato das corridas

Nas eras antigas, havia pré-classificação, grids maiores, carros compartilhados e pilotos que não participavam de todas as etapas. Por isso, uma linha de resultado antiga pode representar uma situação diferente de uma linha recente.

### Comparabilidade com temporadas atuais

Desde 2003, o número de pilotos e equipes, o sistema de pontuação e a estrutura do campeonato se aproximam mais das temporadas usadas na validação e no teste.

### Tamanho da amostra

O recorte ainda reúne milhares de resultados e centenas de corridas. Isso oferece uma base prática para treinar o modelo sem misturar eras com formatos muito diferentes.

## Fins de semana com sprint

Removemos por completo as corridas realizadas em fins de semana com sprint, incluindo o GP principal. Esses eventos seguem uma programação diferente de treinos e classificação. A amostra fica restrita ao formato sem sprint, para que o modelo aprenda e seja avaliado em corridas com uma programação comparável.

## Pit stops e Driver of the Day fora das features

Pit stops podem ter vários registros por piloto e corrida. Um join direto duplicaria linhas; seria preciso resumir as paradas por corrida e piloto. Como elas acontecem durante a prova, também não estão disponíveis para uma previsão feita antes da largada e podem revelar o andamento da corrida.

O Driver of the Day é definido depois da corrida, então não serve como dado de pré-largada. Esse tipo de sessão não entra no resumo, mas a coluna `rd_driver_of_the_day_percentage` ainda pode aparecer no dataset bruto. Ela deve ser removida ao selecionar as features do modelo.
