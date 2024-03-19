package org.example;


import org.refactoringminer.api.GitHistoryRefactoringMiner;
import org.refactoringminer.api.GitService;
import org.refactoringminer.api.Refactoring;
import org.refactoringminer.api.RefactoringHandler;
import org.refactoringminer.rm1.GitHistoryRefactoringMinerImpl;
import org.refactoringminer.util.GitServiceImpl;
import org.eclipse.jgit.lib.Repository;

import java.util.List;
import java.util.Map;

public class Refactorings {

    public void getRefactorings() throws Exception {
        GitHistoryRefactoringMiner miner = new GitHistoryRefactoringMinerImpl();
        GitService gitService = new GitServiceImpl();
        Repository repo = gitService.openRepository("C:\\Users\\indra\\Documents\\Playground\\Misc\\tbd\\rl-template");
        miner.detectAtCommit(repo, "89e8f6cbc07f8d9f370902ee52d15040b1e56d12", new RefactoringHandler() {
            @Override
            public void handle(String commitId, List<Refactoring> refactorings) {
                System.out.println("Refactorings at " + commitId);
                for (Refactoring ref : refactorings) {
                    System.out.println(ref.toString());
                }
            }
        });
    }
}
