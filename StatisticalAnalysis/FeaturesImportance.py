from sklearn.feature_selection import f_classif, mutual_info_classif
class FeaturesImportance:


    def getFeaturesImportance(self, X, Y):
        anova_f, anova_p = f_classif(X, Y)

        mi = mutual_info_classif(X, Y)

        return {"anova_f": anova_f, "anova_p": anova_p, "mi": mi}