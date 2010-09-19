package com.manning.chapter15;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by IntelliJ IDEA. User: tdunning Date: Sep 19, 2010 Time: 2:08:07 PM To change this
 * template use File | Settings | File Templates.
 */
public class BadTokens {
  public static void main(String[] args) throws IOException {
    Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);

    List<File> files = Lists.newArrayList();
    File base = new File("..", "20news-bydate-train");
    for (File newsgroup : base.listFiles()) {
    files.addAll(Arrays.asList(newsgroup.listFiles()));
    }
    System.out.printf("%d training files\n", files.size());

    System.out.println("\n\n50 bad tokens\n");
    File f = files.get(0);
    TokenStream ts = analyzer.tokenStream("text", Files.newReader(f, Charsets.UTF_8));
    ts.addAttribute(TermAttribute.class);
    int k = 0;
    while (ts.incrementToken() && k < 50) {
      System.out.println(ts.toString());
      k++;
    }

    System.out.println("\n\n50 good tokens\n");
    TokenStream ts1 = analyzer.tokenStream("text", Files.newReader(f, Charsets.UTF_8));
    ts1.addAttribute(TermAttribute.class);
    k = 0;
    while (ts1.incrementToken() && k < 50) {
      System.out.println(ts1.getAttribute(TermAttribute.class).term());
      k++;
    }
  }
}
