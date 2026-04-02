# org.openmarkov.learning.algorithm

Implementaciones concretas de algoritmos de aprendizaje de redes bayesianas. Extiende las abstracciones de `learning.core`.

## Algoritmos implementados

| Algoritmo | Tipo | Clase | discriminative | latent vars |
|---|---|---|---|---|
| Hill Climbing | Score+search (greedy) | `HillClimbingAlgorithm` | false | false |
| PC | Constraint-based | `PCAlgorithm` | false | false |
| Naive Bayes | Discriminativo | `NaiveBayesAlgorithm` | true | false |
| KDB | Discriminativo | `KDBAlgorithm` | true | false |
| FANB | Discriminativo | `ForestAugmentedNBAlgorithm` | true | false |
| TAN | Discriminativo | `TreeAugmentedNBAlgorithm` | true | false |
| SNB | Discriminativo | `SelectiveNBAlgorithm` | true | false |
| SPNB | Discriminativo | `SuperParentNBAlgorithm` | true | false |
| EM | Paramétrico | `EMAlgorithm` | false | true |

> **Nota:** `EMAlgorithm` está **deshabilitado** (bucle principal con `while(false)`). Solo aprende parámetros, no estructura.

## Jerarquía de clases

```
LearningAlgorithm (learning.core)
├── ScoreAndSearchAlgorithm           Base para algoritmos score-based; tiene Metric
│   ├── HillClimbingAlgorithm
│   └── DiscriminativeAlgorithm       Base para clasificadores; implementa IDiscriminativeBayes
│       ├── KDBAlgorithm
│       ├── ForestAugmentedNBAlgorithm
│       ├── TreeAugmentedNBAlgorithm
│       ├── SelectiveNBAlgorithm
│       └── SuperParentNBAlgorithm
├── IndependenceRelationsAlgorithm    Base para algoritmos constraint-based
│   └── PCAlgorithm
├── NaiveBayesAlgorithm               (implementa IDiscriminativeBayes directamente)
└── EMAlgorithm
```

## Hill Climbing

Búsqueda local greedy: en cada paso propone el edit (añadir/eliminar/invertir arco) con mayor mejora de score.

```java
// El constructor es detectado por reflexión desde LearningAlgorithmManager
public HillClimbingAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha, Metric metric)
```

- El historial `lastBestEdits` evita reconsiderar edits ya devueltos en el ciclo actual.
- `getBestEdit()` reinicia el historial; `getNextEdit()` continúa desde donde estaba.
- El score viene del `Metric` de `learning.metric` (BIC, AIC, K2, etc.).

## PC Algorithm

Constraint-based en 3 fases:

```
INITIAL_PHASE              → tests de independencia, elimina arcos
HEAD_TO_HEAD_ORIENTATION   → detecta y orienta colisionadores (X→Z←Y)
REMAINING_LINKS_ORIENTATION → orienta arcos restantes para preservar DAG
```

- Usa `IndependenceTester` (interfaz): implementación concreta `CrossEntropyIndependenceTester` (χ²).
- Cachea resultados de tests por `NodePair` para no repetir cálculos.
- `PCEditMotivation` almacena el conjunto de separación y el p-valor.

```java
public PCAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha,
                   IndependenceTester independenceTester, Double significanceLevel)
```

## Algoritmos discriminativos (Naive Bayes y derivados)

Todos implementan `IDiscriminativeBayes`:
- Requieren designar una **variable clase** (root node)
- Estructura: clase → features, con variaciones en las dependencias entre features

`DiscriminativeAlgorithm` usa dos métricas:
- `metric` — información mutua condicional (condicionada a la clase)
- `unconditionedMetric` — información mutua no condicionada (para arcos clase-feature)

Restricciones automáticas aplicadas: `MaxNumParents`, `NoCycle`.

## Añadir un nuevo algoritmo

1. Extender `LearningAlgorithm` (o `ScoreAndSearchAlgorithm` si usa métricas de score)
2. Anotar con `@LearningAlgorithmType`
3. Constructor con firma `(ProbNet, CaseDatabase, Double, ...)` — detectado por reflexión
4. Implementar `getBestEdit`, `getNextEdit`, `getMotivation`
5. Opcionalmente, crear `*ParametersDialog` en subpaquete `gui/`

```java
@LearningAlgorithmType(name = "My Algorithm", discriminative = false, supportsUnobservedVariables = false)
public class MyAlgorithm extends ScoreAndSearchAlgorithm {
    public MyAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha, Metric metric) {
        super(probNet, caseDatabase, metric, alpha);
    }
    // ...
}
```

## Patrón de historial de edits

Todos los algoritmos mantienen listas de edits ya propuestos para no repetirlos en el mismo ciclo:

```
HillClimbing:  lastBestEdits
PC:            lastRemovedEdits, lastOrientationEdits, lastCompoundOrientationEdits
```

## Dependencias

- `org.openmarkov.core`
- `org.openmarkov.learning.core`
- `org.openmarkov.learning.metric`
- `org.openmarkov.learning.gui`
- `org.openmarkov.gui`
- `org.openmarkov.inference` (usado por EM para HuginPropagation)

## Tests

| Clase | Qué prueba |
|---|---|
| `HillClimbingAlgorithmTest` | add/remove/invert, flag onlyPositive, historial con `StubMetric` |
| `PCAlgorithmTest` | skeleton discovery, orientación de colisionadores con `MockIndependenceTester` |
| `EMAlgorithmTests` | aprendizaje paramétrico con variables latentes |
| `CrossEntropyIndependenceTesterTest` | implementación del test χ² |
| `EditHistorySupportTest` | mecanismo de historial |