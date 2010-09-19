package org.apache.mahout.classifier.sgd;

import com.google.common.base.Splitter;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.vectors.ConstantValueEncoder;
import org.apache.mahout.vectors.Dictionary;
import org.apache.mahout.vectors.FeatureVectorEncoder;
import org.apache.mahout.vectors.StaticWordValueEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Reads and trains an adaptive logistic regression model on the 20 newsgroups data.
 * The first command line argument gives the path of the directory holding the training
 * data.  The optional second argument, leakType, defines which classes of features to use.
 * Importantly, leakType controls whether a synthetic date is injected into the data as
 * a target leak and if so, how.
 *
 * The value of leakType % 3 determines whether the target leak is injected according to
 * the following table:
 * <table>
 * <tr><td>0</td><td>No leak injected</td></tr>
 * <tr><td>1</td><td>Synthetic date injected in MMM-yyyy format. This will be a single token and
 * is a perfect target leak since each newsgroup is given a different month</td></tr>
 * <tr><td>2</td><td>Synthetic date injected in dd-MMM-yyyy HH:mm:ss format.  The day varies
 * and thus there are more leak symbols that need to be learned.  Ultimately this is just
 * as big a leak as case 1.</td></tr>
 * </table>
 * Leaktype also determines what other text will be indexed.  If leakType is greater
 * than or equal to 6, then neither headers nor text body will be used for features and the leak is the only
 * source of data.  If leakType is greater than or equal to 3, then subject words will be used as features.
 * If leakType is less than 3, then both subject and body text will be used as features.
 *
 * A leakType of 0 gives no leak and all textual features. 
 */
public class TrainNewsGroups {
  private static final int FEATURES = 10000;
  // 1997-01-15 00:01:00 GMT
  private static final long DATE_REFERENCE = 853286460;
  private static final long MONTH = 30 * 24 * 3600;
  private static final long WEEK = 7 * 24 * 3600;

  private static final Random rand = new Random();

  private static final Splitter ON_COLON = Splitter.on(":");

  private static final String[] leakLabels = {"none", "month-year", "day-month-year"};
  private static final SimpleDateFormat[] df = new SimpleDateFormat[]{
    new SimpleDateFormat(""),
    new SimpleDateFormat("MMM-yyyy"),
    new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss")
  };

  private static final Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);
  private static final FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
  private static final FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");

  public static void main(String[] args) throws IOException {
    File base = new File(args[0]);

    int leakType = 0;
    if (args.length > 1) {
      leakType = Integer.parseInt(args[1]);
    }

    Dictionary newsGroups = new Dictionary();

    encoder.setProbes(2);
    AdaptiveLogisticRegression learningAlgorithm = new AdaptiveLogisticRegression(20, FEATURES, new L1());
    learningAlgorithm.setInterval(200);

    List<File> files = Lists.newArrayList();
    for (File newsgroup : base.listFiles()) {
      newsGroups.intern(newsgroup.getName());
      files.addAll(Arrays.asList(newsgroup.listFiles()));
    }
    System.out.printf("%d training files\n", files.size());

    double averageLL = 0;
    double averageCorrect = 0;

    int k = 0;
    double step = 0;
    int[] bumps = new int[]{1, 2, 5};
    for (File file : permute(files, rand)) {
      String ng = file.getParentFile().getName();
      int actual = newsGroups.intern(ng);

      Vector v = encodeFeatureVector(file, actual, k, leakType);
      learningAlgorithm.train(actual, v);

      State<AdaptiveLogisticRegression.Wrapper> tmp = learningAlgorithm.getBest();
      CrossFoldLearner state = null;
      if (tmp != null) {
        state = tmp.getPayload().getLearner();
      }
      double ll;
      int estimated;
      if (state != null) {
        ll = state.logLikelihood(actual, v);

        double mu = Math.min(k + 1, 200);
        if (Double.isNaN(averageLL)) {
          averageLL = ll;
        } else {
          averageLL = averageLL + (ll - averageLL) / mu;
        }

        Vector p = state.classifyFull(v);
        estimated = p.maxValueIndex();

        int correct = (estimated == actual) ? 1 : 0;
        if (!Double.isNaN(averageCorrect)) {
          averageCorrect = averageCorrect + (correct - averageCorrect) / mu;
        } else {
          averageCorrect = correct;
        }
      } else {
        estimated = 0;
        
        ll = Double.NaN;
        averageLL = Double.NaN;

        averageCorrect = Double.NaN;
      }

      k++;

      int bump = bumps[(int) Math.floor(step) % bumps.length];
      int scale = (int) Math.pow(10, Math.floor(step / bumps.length));
      State<AdaptiveLogisticRegression.Wrapper> best = learningAlgorithm.getBest();
      double maxBeta;
      double nonZeros;
      double positive;
      double norm;
      if (best != null) {
        maxBeta = best.getPayload().getLearner().getModels().get(0).getBeta().aggregate(Functions.MAX, Functions.IDENTITY);
        nonZeros = best.getPayload().getLearner().getModels().get(0).getBeta().aggregate(Functions.PLUS, new UnaryFunction() {
          @Override
          public double apply(double v) {
            return Math.abs(v) > 1e-9 ? 1 : 0;
          }
        });
        positive = best.getPayload().getLearner().getModels().get(0).getBeta().aggregate(Functions.PLUS, new UnaryFunction() {
          @Override
          public double apply(double v) {
            return v > 0 ? 1 : 0;
          }
        });
        norm = best.getPayload().getLearner().getModels().get(0).getBeta().aggregate(Functions.PLUS, Functions.ABS);
      } else {
        maxBeta = Double.NaN;
        nonZeros = 0;
        positive = 0;
        norm = Double.NaN;
      }
      if (k % (bump * scale) == 0) {
        step += 0.25;
        System.out.printf("%.2f\t%.2f\t%.2f\t%.2f\t", maxBeta, nonZeros, positive, norm);
        System.out.printf("%d\t%.3f\t%.3f\t%.2f\t%s\t%s\t%s\n",
          k, ll, averageLL, averageCorrect * 100, ng,
          newsGroups.values().get(estimated),
          leakLabels[leakType % 3]);
      }
    }
    learningAlgorithm.close();

    dissect(leakType, newsGroups, learningAlgorithm, files, k);
  }

  private static void dissect(int leakType, Dictionary newsGroups, AdaptiveLogisticRegression learningAlgorithm, List<File> files, int k) throws IOException {
    Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();

    ModelDissector md = new ModelDissector(learningAlgorithm.getBest().getPayload().getLearner().numCategories());

    encoder.setTraceDictionary(traceDictionary);
    bias.setTraceDictionary(traceDictionary);
    for (File file : permute(files, rand).subList(0, 500)) {
      String ng = file.getParentFile().getName();
      int actual = newsGroups.intern(ng);

      traceDictionary.clear();
      Vector v = encodeFeatureVector(file, actual, k, leakType);
      md.update(v, traceDictionary, learningAlgorithm.getBest().getPayload().getLearner());
    }

    List<String> ngNames = Lists.newArrayList(newsGroups.values());
    List<ModelDissector.Weight> weights = md.summary(100);
    for (ModelDissector.Weight w : weights) {
      System.out.printf("%s\t%.1f\t%s\n", w.getFeature(), w.getWeight(), ngNames.get(w.getMaxImpact() + 1));
    }

  }

  private static Vector encodeFeatureVector(File file, int actual, int recordNumber, int leakType) throws IOException {
    long date = (long) (1000 * (DATE_REFERENCE + actual * MONTH + 1 * WEEK * rand.nextDouble()));
    Multiset<String> words = ConcurrentHashMultiset.create();

    BufferedReader reader = new BufferedReader(new FileReader(file));
    String line = reader.readLine();
    Reader dateString = new StringReader(df[leakType % 3].format(new Date(date)));
    countWords(analyzer, words, dateString);
    while (line != null && line.length() > 0) {
      boolean countHeader = (
        line.startsWith("From:") || line.startsWith("Subject:") ||
          line.startsWith("Keywords:") || line.startsWith("Summary:")) && (leakType < 6);
      do {
        StringReader in = new StringReader(line);
        if (countHeader) {
          countWords(analyzer, words, in);
        }
        line = reader.readLine();
      } while (line.startsWith(" "));
    }
    if (leakType < 3) {
      countWords(analyzer, words, reader);
    }
    reader.close();

    Vector v = new RandomAccessSparseVector(FEATURES);
    bias.addToVector("", 1, v);
    for (String word : words.elementSet()) {
      encoder.addToVector(word, Math.log(1 + words.count(word)), v);
    }
    return v;
  }

  private static void countWords(Analyzer analyzer, Multiset<String> words, Reader in) throws IOException {
    TokenStream ts = analyzer.tokenStream("text", in);
    ts.addAttribute(TermAttribute.class);
    while (ts.incrementToken()) {
      String s = ts.getAttribute(TermAttribute.class).term();
      words.add(s);
    }
  }

  private static List<File> permute(List<File> files, Random rand) {
    List<File> r = Lists.newArrayList();
    for (File file : files) {
      int i = rand.nextInt(r.size() + 1);
      if (i == r.size()) {
        r.add(file);
      } else {
        r.add(r.get(i));
        r.set(i, file);
      }
    }
    return r;
  }

}