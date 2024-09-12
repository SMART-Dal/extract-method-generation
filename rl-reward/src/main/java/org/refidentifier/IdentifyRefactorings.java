package org.refidentifier;

import org.eclipse.jgit.lib.Repository;
import org.refactoringminer.api.GitHistoryRefactoringMiner;
import org.refactoringminer.api.Refactoring;
import org.refactoringminer.api.RefactoringHandler;
import org.refactoringminer.api.RefactoringType;
import org.refactoringminer.rm1.GitHistoryRefactoringMinerImpl;

import java.util.List;

public class IdentifyRefactorings {

    private final Repository repo;
    private final String startCommit;
    private final String endCommit;
    private int c ;

    public IdentifyRefactorings(Repository repository, String startCommit, String endCommit) {
        this.repo = repository;
        this.startCommit = startCommit;
        this.endCommit = endCommit;
        this.c = 0;
    }
    public boolean identifyRefactoringInstances() throws Exception {
        GitHistoryRefactoringMiner miner = new GitHistoryRefactoringMinerImpl();
        miner.detectBetweenCommits(repo, startCommit, endCommit, new RefactoringHandler(){
            @Override
            public void handle(String commitId, List<Refactoring> refactorings) {
                for (Refactoring ref : refactorings) {
                    if (ref.getRefactoringType() == RefactoringType.EXTRACT_OPERATION){
                        c++;
                    }
                }
            }
        });
        return c > 0;
    }
}
