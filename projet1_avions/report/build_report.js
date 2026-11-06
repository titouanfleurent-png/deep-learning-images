// Génère le rapport et la fiche d'oral du projet 1. Usage : node build_report.js (depuis ce dossier)
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, Footer, PageNumber, ImageRun } = require(path.join(__dirname, '../../_journal/node_modules/docx'));

const ACCENT = '1F4E79', W = 9026, FIG = path.join(__dirname, '../results/figures');
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const p = t => new Paragraph({ spacing: { after: 120 }, children: [new TextRun(t)] });
const li = t => new Paragraph({ numbering: { reference: 'bul', level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });
const cell = (t, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA },
  shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
  margins: { top: 50, bottom: 50, left: 90, right: 90 },
  children: [new Paragraph({ alignment: o.center ? AlignmentType.CENTER : undefined, children: [new TextRun({ text: String(t), bold: !!o.bold, color: o.color, size: 19 })] })] });
const table = (widths, head, rows) => new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [
  new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, widths[i], { bold: true, fill: ACCENT, color: 'FFFFFF', center: i > 0 })) }),
  ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, widths[i], { center: i > 0, bold: i === 0 })) })),
] });
const caption = t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 180 }, children: [new TextRun({ text: t, italics: true, size: 18, color: '595959' })] });
const fig = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
  children: [new ImageRun({ type: path.extname(file).slice(1).replace('jpeg', 'jpg'), data: fs.readFileSync(path.join(FIG, file)), transformation: { width: w, height: h } })] });
const title = (t, s) => [new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(t)] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: s, italics: true, color: '595959' })] })];
const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

const report = [
  ...title("Reconnaissance d'avions en hiérarchie", 'Projet 1, cours de deep learning appliqué aux images. Titouan Fleurent, CNAM, novembre 2026.'),
  h1('Résumé'),
  p("Reconnaître un avion à la variante près est difficile, et toutes les erreurs n'ont pas la même gravité : confondre un 737-800 avec un 737-900 est moins grave que le confondre avec un A320. Ce projet teste si l'on peut réduire la gravité des erreurs en donnant au réseau la hiérarchie constructeur, famille, variante."),
  p("Trois modèles EfficientNet-B0 pré-entraînés sont comparés sur FGVC-Aircraft, sur 3 graines chacun. Sur le test, les deux modèles hiérarchiques réduisent les erreurs graves (autre constructeur) d'environ 2 à 2,5 points, soit 14 à 16 % de moins que la baseline, sans perdre en accuracy sur les variantes. Ils ne se distinguent pas l'un de l'autre sur ces critères, mais le modèle à soft labels produit des prédictions nettement plus cohérentes entre les niveaux."),
  h1('1. Problématique'),
  p("La question est la suivante : un CNN qui exploite la hiérarchie des classes fait-il des erreurs moins graves, et à quel prix sur l'accuracy ? Elle prolonge deux articles : Maji et al. (2013), qui introduisent le jeu de données, et Bertinetto et al. (2020), qui proposent de juger un modèle sur la gravité de ses erreurs, pas seulement sur leur nombre."),
  h1('2. Données'),
  li("FGVC-Aircraft : 10 000 photos, 30 constructeurs, 70 familles, 100 variantes, environ 100 images par variante."),
  li("Splits officiels : 3 334 images en entraînement, 3 333 en validation, 3 333 en test. L'entraînement n'utilise que le split train."),
  li("Un bandeau de 20 px contenant le nom du photographe est retiré en premier. Il pourrait servir de raccourci au modèle, car un photographe peut être associé à certains types d'avions."),
  li("Images redimensionnées en carré de 257 px sans recadrage préalable, pour ne pas couper le nez ni la queue, puis recadrées à 224 px : aléatoirement à l'entraînement, au centre à l'évaluation."),
  li("Augmentations à l'entraînement : recadrage aléatoire, miroir horizontal et variations de 20 % de luminosité, contraste et saturation. Normalisation ImageNet."),
  h1('3. Méthode'),
  h2('3.1 Les trois modèles'),
  p("Les trois modèles partagent le même backbone, EfficientNet-B0 pré-entraîné sur ImageNet, et les mêmes réglages d'entraînement."),
  table([1900, 2300, 4826], ['Modèle', 'Têtes', 'Fonction de perte'], [
    ['Plat', '1 (100 variantes)', 'Cross-entropy, label smoothing 0,1'],
    ['Hiérarchique', '3 (30, 70, 100)', 'CE variante + 0,5 x (CE constructeur + CE famille)'],
    ['Soft labels', '1 (100 variantes)', 'Cross-entropy vers une cible douce : softmax(-10 x hauteur / 3)'],
  ]), spacer(),
  p("La hauteur est celle du plus proche ancêtre commun entre deux variantes : 0 pour la même variante, 1 pour la même famille, 2 pour le même constructeur, 3 pour des constructeurs différents. Avec beta = 10, la bonne classe garde un poids de 0,787 dans la cible."),
  p("Pour les modèles à une seule tête, les probabilités de famille et de constructeur sont obtenues en sommant celles des variantes correspondantes."),
  h2('3.2 Entraînement'),
  li("3 epochs avec le backbone gelé, BatchNorm comprises, puis 20 epochs de fine-tuning."),
  li("AdamW, taux d'apprentissage de 1e-4 pour le backbone et 1e-3 pour les têtes, weight decay 1e-4, décroissance en cosinus, batch de 32."),
  li("Environ 85 à 105 s par epoch sur MacBook Air M4 (GPU MPS), soit 9 entraînements en un peu moins de 5 h."),
  h2('3.3 Protocole d\'évaluation'),
  li("Le meilleur checkpoint de chaque run est choisi sur l'accuracy variante en validation."),
  li("Le protocole a été écrit et figé avant de lancer le test (results/protocole_fige.md). Le test n'a été évalué qu'une seule fois."),
  li("Résultats donnés en moyenne et écart-type sur 3 graines."),
  li("Différences testées par bootstrap apparié sur les images (2 000 tirages, IC à 95 %), avec les prédictions moyennées sur les 3 graines par image."),
  h1('4. Résultats sur le test'),
  table([3626, 1800, 1800, 1800], ['Métrique', 'Plat', 'Hiérarchique', 'Soft labels'], [
    ['Accuracy variante (%)', '68,7 ± 0,2', '69,6 ± 0,6', '69,6 ± 0,4'],
    ['Accuracy famille (%)', '79,8 ± 0,4', '81,8 ± 0,2', '80,7 ± 0,5'],
    ['Accuracy constructeur (%)', '86,4 ± 0,1', '85,9 ± 0,1', '87,3 ± 0,4'],
    ['Top-5 variante (%)', '89,7 ± 0,3', '90,7 ± 0,2', '91,4 ± 0,1'],
    ['Gravité moyenne, toutes images', '0,676 ± 0,002', '0,622 ± 0,011', '0,626 ± 0,010'],
    ["Gravité moyenne d'une erreur", '2,16 ± 0,01', '2,05 ± 0,01', '2,06 ± 0,01'],
    ['Erreurs graves (% des images)', '14,9 ± 0,2', '12,4 ± 0,3', '12,8 ± 0,3'],
    ['Cohérence des 3 niveaux (%)', '91,9 ± 0,3', '83,7 ± 0,1', '95,4 ± 0,1'],
  ]),
  caption("Tableau 1. Test, moyenne ± écart-type sur 3 graines. Une erreur tirée au hasard aurait une gravité de 2,91."),
  fig('test_gravite.png', 470, 204),
  caption("Figure 1. Répartition des images selon la gravité de la prédiction (test, moyenne sur 3 graines)."),
  table([2626, 2400, 2000, 2000], ['Différence', 'Accuracy variante (pts)', 'Gravité moyenne', 'Erreurs graves (pts)'], [
    ['Hiérarchique - Plat', '+0,95 [0,23 ; 1,60]', '-0,054 [-0,070 ; -0,037]', '-2,44 [-2,99 ; -1,88]'],
    ['Soft - Plat', '+0,89 [0,36 ; 1,45]', '-0,050 [-0,064 ; -0,036]', '-2,13 [-2,66 ; -1,59]'],
    ['Soft - Hiérarchique', '-0,06 [-0,77 ; 0,68]', '+0,004 [-0,013 ; 0,020]', '+0,31 [-0,26 ; 0,91]'],
  ]),
  caption("Tableau 2. Différences sur le test, avec IC à 95 % par bootstrap apparié."),
  h2('Lecture'),
  li("Les deux modèles hiérarchiques réduisent les erreurs graves par rapport au plat, soit 16 % de moins pour le modèle à 3 têtes et 14 % pour les soft labels. L'intervalle exclut zéro dans les deux cas."),
  li("Cette baisse n'a pas de coût sur l'accuracy variante, qui gagne même environ 1 point. Ce gain reste modeste, avec un intervalle proche de zéro pour le modèle à 3 têtes."),
  li("Les deux approches hiérarchiques ne se distinguent pas sur les 3 critères testés : tous les intervalles contiennent zéro."),
  li("Elles se distinguent ailleurs. Le modèle à 3 têtes se contredit sur 16 % des images, car ses têtes sont indépendantes. Les soft labels atteignent 95 % de cohérence, avec une seule tête. Ils sont donc retenus pour la démo. Ce choix avait été fait sur la validation, avant le test."),
  li("Le modèle à 3 têtes obtient une accuracy constructeur légèrement inférieure au plat sur le test (85,9 % contre 86,4 %), alors qu'elle était supérieure en validation (86,4 % contre 85,7 %). Son avantage au niveau constructeur n'est donc pas établi."),
  fig('courbes_validation.png', 420, 250),
  caption("Figure 2. Accuracy variante en validation au fil des epochs. Le gain le plus net arrive dès le dégel du backbone."),
  h1("5. Analyse d'erreurs"),
  table([2600, 2600, 1913, 1913], ['Variante réelle', 'Variante prédite', 'Images (sur ~33)', 'Hauteur'], [
    ['C-47', 'DC-3', '15,3', '2'], ['E-190', 'E-195', '14,0', '1'], ['MD-11', 'DC-10', '11,7', '2'], ['DC-3', 'C-47', '11,3', '2'],
    ['A340-200', 'A340-300', '10,7', '1'], ['BAE 146-300', 'BAE 146-200', '10,3', '1'],
  ]),
  caption("Tableau 3. Confusions les plus fréquentes (soft labels, test, moyenne sur 3 graines)."),
  li("Les confusions les plus fréquentes portent sur des avions proches : variantes de longueur différente ou appareils dérivés l'un de l'autre, comme le C-47, version militaire du DC-3."),
  li("La hiérarchie du jeu de données a ses limites : C-47 et DC-3 sont dans deux familles distinctes, donc cette confusion compte en hauteur 2 alors que les avions sont très proches."),
  li("Les variantes les plus difficiles sont 737-300 (26 %), 767-300 (33 %), C-47 (36 %), 737-800 (38 %), 747-200 (38 %) et A320 (38 %). Les mieux reconnues sont des appareils au profil unique : Il-76, Eurofighter Typhoon et SR-20 (97 %)."),
  li("Les différences entre variantes tiennent souvent à la longueur du fuselage ou au nombre de hublots, deux détails difficiles à voir à 224 px. C'est une hypothèse, qu'un entraînement à plus haute résolution permettrait de tester."),
  h1('6. Grad-CAM'),
  fig('gradcam_planche.png', 480, 368),
  caption("Figure 3. Images de validation tirées au hasard. Ligne 1 : bonne variante. Ligne 2 : même famille. Ligne 3 : autre constructeur."),
  p("Grad-CAM montre les zones qui ont le plus influencé la classe prédite. Sur cette planche, les bonnes réponses et les erreurs de même famille s'appuient presque toujours sur l'avion. Pour les erreurs graves, l'activation se porte plusieurs fois sur le sol ou l'arrière-plan (PA-28, DHC-8). Le modèle a donc peut-être été gêné par le contexte sur ces images."),
  p("Cette lecture reste qualitative : elle repose sur 12 images et ne prouve pas de lien de cause à effet. Une mesure rigoureuse demanderait des masques de segmentation des avions."),
  h1('7. Démonstration'),
  p("Une application Gradio (app.py) prend une photo et affiche le top-3 à chaque niveau ainsi que la carte Grad-CAM. Elle a été testée en local sur un exemple de validation : A380 reconnu à 95 %."),
  fig('demo_gradio.jpg', 240, 232),
  caption("Figure 4. Démo locale."),
  h1('8. Limites'),
  li("Une seule architecture et une seule résolution ont été testées. L'essai prévu avec ConvNeXt-Tiny n'a pas été fait : il ne répondait pas directement à la question posée, qui porte sur la hiérarchie."),
  li("Les poids w_coarse = 0,5 et beta = 10 n'ont pas été optimisés. Un réglage différent pourrait changer le compromis entre les modèles."),
  li("Le bootstrap porte sur les images : il ne couvre pas toute la variabilité entre graines, et 3 graines donnent un écart-type peu précis. Aucune correction pour comparaisons multiples n'est appliquée sur les 9 intervalles."),
  li("La généralisation à des photos prises hors du jeu de données n'a pas été testée. Les scores annoncés ne valent que pour ce type d'images."),
  li("La gravité dépend de la hiérarchie du jeu de données, qui ne reflète pas toujours la proximité réelle des avions."),
  h1('9. Conclusion'),
  p("Sur FGVC-Aircraft, intégrer la hiérarchie au modèle réduit les erreurs graves de 14 à 16 % sans coût sur l'accuracy. La méthode la plus simple est aussi la plus satisfaisante : changer uniquement la cible d'apprentissage avec des soft labels donne le même gain qu'un modèle à 3 têtes, avec des prédictions plus cohérentes."),
  p("Pistes : tester une résolution plus élevée pour les variantes proches, régler beta et w_coarse sur la validation, et évaluer le modèle sur des photos prises hors du jeu de données."),
  h1('Références'),
  li("Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A. (2013). Fine-Grained Visual Classification of Aircraft. arXiv:1306.5151."),
  li("Bertinetto, L., Mueller, R., Tertikas, K., Samangooei, S., Lord, N. A. (2020). Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks. CVPR."),
  li("Tan, M., Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML."),
  li("Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV."),
];

const qa = (q, a) => [new Paragraph({ spacing: { before: 140, after: 40 }, keepNext: true, children: [new TextRun({ text: q, bold: true })] }), p(a)];
const oral = [
  ...title("Fiche d'oral : projet avions", 'Les choix à savoir justifier, avec une réponse courte pour chacun.'),
  h1('Le message en 30 secondes'),
  p("En donnant au réseau la hiérarchie des avions, on réduit les erreurs graves de 14 à 16 % sans perdre d'accuracy. Changer uniquement les cibles avec des soft labels suffit : même gain qu'un modèle à 3 têtes, avec des prédictions plus cohérentes."),
  h1('Données'),
  ...qa("Pourquoi retirer le bandeau de 20 px ?", "Il contient le nom du photographe. Le modèle pourrait l'utiliser comme raccourci plutôt que de regarder l'avion."),
  ...qa("Pourquoi redimensionner en carré, au risque de déformer ?", "Les photos sont larges. Un recadrage carré couperait le nez et la queue, qui sont discriminants. Une légère déformation coûte moins cher."),
  ...qa("Pourquoi pas de miroir vertical ?", "Un avion n'est presque jamais à l'envers sur ces photos. Le miroir horizontal, lui, reste réaliste."),
  ...qa("Pourquoi la normalisation ImageNet ?", "Le backbone a été pré-entraîné avec ces statistiques, il faut lui fournir des entrées comparables."),
  h1('Modèle et entraînement'),
  ...qa("Pourquoi le transfer learning ?", "Avec environ 33 images d'entraînement par variante, apprendre un CNN de zéro serait voué au surapprentissage. ImageNet apporte déjà des filtres généraux."),
  ...qa("Pourquoi EfficientNet-B0 ?", "Bon compromis entre précision et coût, adapté à un MacBook Air sans ventilateur. La question porte sur la hiérarchie, pas sur le meilleur backbone."),
  ...qa("Pourquoi 3 epochs de gel, BatchNorm comprises ?", "Les têtes partent de valeurs aléatoires : leurs gradients perturberaient un backbone déjà bon. On gèle aussi les BatchNorm, sinon leurs statistiques continueraient à bouger."),
  ...qa("Pourquoi deux taux d'apprentissage ?", "1e-3 pour les têtes, qui ont tout à apprendre, et 1e-4 pour le backbone, qu'on ajuste seulement."),
  ...qa("Comment fonctionnent les soft labels ?", "La cible n'est plus un seul 1. On applique un softmax à -beta x hauteur / 3, ce qui donne du poids aux classes proches. Avec beta = 10, la bonne classe garde 0,787. Plus beta est grand, plus on se rapproche d'une cible classique."),
  ...qa("Pourquoi w_coarse = 0,5 ?", "Pour garder la variante comme objectif principal tout en régularisant avec les niveaux supérieurs. Cette valeur n'a pas été optimisée, c'est une limite."),
  ...qa("Comment comparer à 3 niveaux un modèle à une seule tête ?", "Par marginalisation : la probabilité d'une famille est la somme des probabilités de ses variantes."),
  h1('Évaluation'),
  ...qa("Qu'est-ce que la gravité d'une erreur ?", "La hauteur du plus proche ancêtre commun entre vraie classe et prédiction : 1 même famille, 2 même constructeur, 3 autre constructeur. Une erreur au hasard vaudrait 2,91."),
  ...qa("Pourquoi deux gravités, 2,06 et 0,63 pour le même modèle ?", "2,06 est la moyenne sur les seules erreurs. 0,63 compte les bonnes réponses à 0 : elle tient donc compte du nombre d'erreurs et de leur gravité. Un modèle pourrait améliorer la première en faisant simplement plus d'erreurs bénignes."),
  ...qa("Pourquoi 3 graines et un bootstrap ?", "Les graines mesurent la variabilité de l'entraînement, le bootstrap celle due à l'échantillon d'images. Il est apparié car les modèles sont comparés sur les mêmes images."),
  ...qa("Comment le test a-t-il été protégé ?", "Tous les choix ont été faits sur la validation, le protocole a été écrit avant, et le test n'a été lancé qu'une fois."),
  ...qa("Pourquoi la cohérence du plat n'est-elle pas de 100 % ?", "La famille prédite est celle dont la somme de probabilités est la plus forte. Elle peut différer de la famille de la variante la plus probable, si les probabilités sont réparties entre plusieurs variantes."),
  h1('Questions pièges'),
  ...qa("Vos 70 % ne sont-ils pas faibles ?", "Il y a 100 classes très proches et environ 33 images d'entraînement par classe, à 224 px. Le hasard donnerait 1 %. Une résolution ou un backbone plus élevés amélioreraient sans doute ce score, mais ce n'était pas la question du projet."),
  ...qa("Le gain d'accuracy d'environ 1 point est-il réel ?", "L'intervalle exclut zéro mais reste près de zéro pour le modèle à 3 têtes (0,23 à 1,60). Le résultat solide concerne les erreurs graves, pas l'accuracy."),
  ...qa("Grad-CAM prouve-t-il que le modèle regarde les bons détails ?", "Non. C'est un outil qualitatif, montré sur 12 images tirées au hasard. Il aide à formuler des hypothèses, comme l'influence de l'arrière-plan sur certaines erreurs graves."),
  ...qa("Pourquoi le modèle à 3 têtes n'est-il pas retenu ?", "Il ne fait pas mieux sur les critères testés, se contredit sur 16 % des images et a plus de paramètres."),
  ...qa("Que feriez-vous avec plus de temps ?", "Tester 448 px, régler beta et w_coarse sur la validation, et évaluer sur des photos hors du jeu de données."),
];

function build(children, file, docTitle) {
  const doc = new Document({ creator: 'Titouan Fleurent', title: docTitle,
    styles: { default: { document: { run: { font: 'Calibri', size: 22 } },
      title: { run: { font: 'Calibri', size: 40, bold: true, color: ACCENT }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      heading1: { run: { font: 'Calibri', size: 28, bold: true, color: ACCENT }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 2 } } } },
      heading2: { run: { font: 'Calibri', size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true } } } },
    numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] }] },
    sections: [{ properties: { page: { margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '808080' })] })] }) }, children }] });
  return Packer.toBuffer(doc).then(b => fs.writeFileSync(path.join(__dirname, file), b));
}
build(report, 'Rapport_projet1_avions.docx', "Reconnaissance d'avions en hiérarchie").then(() => build(oral, 'Fiche_oral_projet1.docx', "Fiche d'oral")).then(() => console.log('ok'));
