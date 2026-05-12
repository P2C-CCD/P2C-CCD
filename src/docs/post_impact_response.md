## Post-Impact Response

descriptionaddeddescription post-impact response layer, Entry pointin:

- `python/p2cccd/data/response.py`

descriptionis notreplace CCD detection, descriptionis notdescription benchmark originalInputtrajectory, instead:

1. indescriptioncollisiondescriptionanddescriptionhas `TOI` descriptionunder, constructdescription**collisionafterdescription replay**.
2. invisualizationanddescriptiongenerate case in, by**momentumdescription**and**coefficient of restitution**generate `BounceReplay`.
3. keep external benchmark  raw motion, ground truth, `FN` and `Recall` statisticsProtocoldescription.

### physics model

current response Modeldescriptionuse:

- description
- description
- description
- contactdescription TOI whendescriptionindescription
- defaultcoefficient of restitution `e = 1.0`

description:

```text
j = - (1 + e) * ((v_a - v_b) - n) / (1 / m_a + 1 / m_b)
v_a' = v_a + (j / m_a) * n
v_b' = v_b - (j / m_b) * n
```

among them:

- `n` fromdescription A description B
- `m_a / m_b` isquality
- `v_a / v_b` iscollisiondescription

### qualitydescription proxy or mesh replay, currentdescriptionuse**descriptionqualitydescription**:

```text
m = 4 / 3 * pi * r^3
```

descriptionguarantee:

- all internal sample descriptionhasdescriptionquality
- real-mesh visualizationdescriptionfrom centered mesh  bounding radius descriptionquality
- descriptionanddescription

### descriptionconnectdescription connectdescription:

1. `MotionDiscPairSample`
   - description `mass_a`, `mass_b`, `restitution`

2. internal analytic samplers
   - default generation quality

3. ABC CAD proxy samples
   - default generation quality

4. `abc_mesh_exact_views.py`
   - real mesh descriptionsupport `BounceReplay`
   - default mode is `BounceReplay`

5. `cad_proxy_views.py`
   - addeddescription CAD proxy collision animation
   - default mode is `BounceReplay`

### asdescriptionconnectdescription external benchmark query

ifdescriptionconnect external benchmark original query description"collisionafterdescription"splitdescriptiontrajectory, description:

- original benchmark description
- query  ground-truth description
- `candidate_recall` and `final FN` statisticsdescription

thereforecurrentdescriptionis:

- **raw benchmark query keepdescription**
- **response as replay / visualization layerand internal-case layerdescription**

descriptioniscurrentdescriptionin correctness and physics replay description.
