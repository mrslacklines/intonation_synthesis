VUV = [
    'vuv',
]

QUINTPHONE_SEGMENT_VC_LL = [
    "QS LL-Vowel{i^*,y^*,e^*,a^*,o^*,u^*,schwa^*}",
    "QS LL-Consonant{gs^*,p^*,b^*,t^*,d^*,k^*,g^*,ki^*,gi^*,f^*,v^*,s^*,si^*,z^*,zi^*,sz^*,rz^*,x^*,c^*,dz^*,cz^*,drz^*,ci^*,dzi^*,m^*,n^*,ni^*,ng^*,l^*,r^*,w^*,ww^*,j^*,jj^*}",
]
QUINTPHONE_SEGMENT_VC_L = [
    "QS L-Vowel{*^schwa-,*^i-,*^y-,*^e-,*^a-,*^o-,*^u-}",
    "QS L-Consonant                       {*^gs-,*^p-,*^b-,*^t-,*^d-,*^k-,*^g-,*^ki-,*^gi-,*^f-,*^v-,*^s-,*^si-,*^z-,*^zi-,*^sz-,*^rz-,*^x-,*^c-,*^dz-,*^cz-,*^drz-,*^ci-,*^dzi-,*^m-,*^n-,*^ni-,*^ng-,*^l-,*^r-,*^w-,*^ww-,*^j-,*^jj-}",
]
QUINTPHONE_SEGMENT_VC_C = [
    "QS C-Vowel{*-schwa+*,*-i+*,*-y+*,*-e+*,*-a+*,*-o+*,*-u+*}",
    "QS C-Consonant                       {*-gs+*,*-p+*,*-b+*,*-t+*,*-d+*,*-k+*,*-g+*,*-ki+*,*-gi+*,*-f+*,*-v+*,*-s+*,*-si+*,*-z+*,*-zi+*,*-sz+*,*-rz+*,*-x+*,*-c+*,*-dz+*,*-cz+*,*-drz+*,*-ci+*,*-dzi+*,*-m+*,*-n+*,*-ni+*,*-ng+*,*-l+*,*-r+*,*-w+*,*-ww+*,*-j+*,*-jj+*}",
]
QUINTPHONE_SEGMENT_VC_R = [
    "QS R-Vowel{*+i=*,*+y=*,*+e=*,*+a=*,*+o=*,*+u=*,*+schwa=*}",
    "QS R-Consonant                       {*+gs=*,*+p=*,*+b=*,*+t=*,*+d=*,*+k=*,*+g=*,*+ki=*,*+gi=*,*+f=*,*+v=*,*+s=*,*+si=*,*+z=*,*+zi=*,*+sz=*,*+rz=*,*+x=*,*+c=*,*+dz=*,*+cz=*,*+drz=*,*+ci=*,*+dzi=*,*+m=*,*+n=*,*+ni=*,*+ng=*,*+l=*,*+r=*,*+w=*,*+ww=*,*+j=*,*+jj=*}",
]
QUINTPHONE_SEGMENT_VC_RR = [
    "QS RR-Vowel{*=i@*,*=y@*,*=e@*,*=a@*,*=o@*,*=u@*,*=schwa@*}",
    "QS RR-Consonant                       {*=gs@*,*=p@*,*=b@*,*=t@*,*=d@*,*=k@*,*=g@*,*=ki@*,*=gi@*,*=f@*,*=v@*,*=s@*,*=si@*,*=z@*,*=zi@*,*=sz@*,*=rz@*,*=x@*,*=c@*,*=dz@*,*=cz@*,*=drz@*,*=ci@*,*=dzi@*,*=m@*,*=n@*,*=ni@*,*=ng@*,*=l@*,*=r@*,*=w@*,*=ww@*,*=j@*,*=jj@*}",
]

QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_LL = [
    "QS LL-Stop{gs^*,p^*,b^*,t^*,d^*,k^*,g^*}",
    "QS LL-Nasal{ww^*,jj^*,m^*,n^*,ni^*,ng^*}",
    "QS LL-Fricative{f^*,v^*,s^*,si^*,z^*,zi^*,sz^*,rz^*,x^*}",
    "QS LL-Front{e^*,i^*,y^*,f^*,v^*,p^*,b^*,m^*,w^*,ww^*}",
    "QS LL-Central{schwa^*,a^*,t^*,d^*,s^*,si^*,z^*,zi^*,n^*,r^*,l^*,t^*,d^*,sz^*,rz^*,cz^*,drz^*,c^*,dz^*,ci^*,dzi^*}",
    "QS LL-Back{o^*,u^*,k^*,g^*,ki^*,gi^*,ng^*,x^*,gs^*}",
    "QS LL-Front_Vowel{e^*,i^*,y^*}",
    "QS LL-Central_Vowel{a^*,schwa^*}",
    "QS LL-Back_Vowel{o^*,u^*}",
    "QS LL-High_Vowel{i^*,y^*,u^*}",
    "QS LL-Medium_Vowel{e^*,o^*}",
    "QS LL-Low_Vowel{a^*}",
    "QS LL-Rounded_Vowel{o^*,u^*}",
    "QS LL-Unrounded_Vowel{a^*,e^*,i^*,y^*}",
    "QS LL-IVowel{i^*}",
    "QS LL-EVowel{e^*}",
    "QS LL-AVowel{a^*}",
    "QS LL-OVowel{o^*}",
    "QS LL-UVowel{u^*}",
    "QS LL-YVowel                          {y^*}",
    "QS LL-SCHWAVowel                      {schwa^*}",
    "QS LL-Unvoiced_Consonant{gs^*,p^*,t^*,k^*,ki^*,f^*,v^*,s^*,sz^*,x^*,c^*,cz^*,ci^*}",
    "QS LL-Voiced_Consonant{b^*,d^*,g^*,gi^*,v^*,z^*,zi^*,rz^*,dz^*,drz^*,dzi^*,m^*,n^*,ni^*,ng^*,l^*,r^*,w^*,ww^*,j^*,jj^*}",
    "QS LL-Front_Consonant{f^*,v^*,f^*,p^*,b^*,m^*,w^*,ww^*}",
    "QS LL-Central_Consonant{t^*,d^*,s^*,si^*,z^*,zi^*,n^*,r^*,l^*,t^*,d^*,sz^*,rz^*,cz^*,drz^*,c^*,dz^*,ci^*,dzi^*}",
    "QS LL-Back_Consonant{gs^*,k^*,g^*,ki^*,gi^*,ng^*,x^*}",
    "QS LL-Fortis_Consonant{gs^*,cz^*,f^*,k^*,p^*,s^*,sz^*,t^*,ci^*,c^*,ki^*}",
    "QS LL-Lenis_Consonant{drz^*,v^*,g^*,b^*,rz^*,z^*,d^*,dzi^*,dz^*,gi^*,zi^*}",
    "QS LL-Neigther_F_or_L{m^*,n^*,ni^*,ng^*,l^*,r^*,w^*,ww^*,j^*,jj^*}",
    "QS LL-Voiced_Stop{b^*,d^*,g^*}",
    "QS LL-Unvoiced_Stop{p^*,t^*,k^*,gs^*}",
    "QS LL-Front_Stop{b^*,p^*}",
    "QS LL-Central_Stop{d^*,t^*}",
    "QS LL-Back_Stop{g^*,k^*,gs^*}",
    "QS LL-Voiced_Fricative{v^*,z^*,zi^*,rz^*}",
    "QS LL-Unvoiced_Fricative{f^*,s^*,si^*,sz^*,x^*}",
    "QS LL-Front_Fricative{f^*,v^*}",
    "QS LL-Affricate_Consonant{dz^*,drz^*,dzi^*,c^*,cz^*,ci^*}",
    "QS LL-silences{pau^*}",
    "QS LL-schwa                                   {schwa^*}",
    "QS LL-a{a^*}",
    "QS LL-e{e^*}",
    "QS LL-i{i^*}",
    "QS LL-y{y^*}",
    "QS LL-o{o^*}",
    "QS LL-u{u^*}",
    "QS LL-p{p^*}",
    "QS LL-b{b^*}",
    "QS LL-t{t^*}",
    "QS LL-d{d^*}",
    "QS LL-k{k^*}",
    "QS LL-ki{ki^*}",
    "QS LL-g{g^*}",
    "QS LL-gi{gi^*}",
    "QS LL-f{f^*}",
    "QS LL-v{v^*}",
    "QS LL-s{s^*}",
    "QS LL-si{si^*}",
    "QS LL-z{z^*}",
    "QS LL-zi{zi^*}",
    "QS LL-sz{sz^*}",
    "QS LL-rz{rz^*}",
    "QS LL-x{x^*}",
    "QS LL-c{c^*}",
    "QS LL-dz{dz^*}",
    "QS LL-cz{cz^*}",
    "QS LL-drz{drz^*}",
    "QS LL-ci{ci^*}",
    "QS LL-dzi{dzi^*}",
    "QS LL-m{m^*}",
    "QS LL-n{n^*}",
    "QS LL-ni{ni^*}",
    "QS LL-ng{ng^*}",
    "QS LL-l{l^*}",
    "QS LL-r{r^*}",
    "QS LL-w{w^*}",
    "QS LL-ww{ww^*}",
    "QS LL-j{j^*}",
    "QS LL-jj{jj^*}",
    "QS LL-gs                                      {gs^*}",
]
QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_L = [
    "QS L-Stop{*^p-,*^b-,*^t-,*^d-,*^k-,*^g-,*^gs-}",
    "QS L-Nasal{*^ww-,*^jj-,*^m-,*^n-,*^ni-,*^ng-}",
    "QS L-Fricative{*^f-,*^v-,*^s-,*^si-,*^z-,*^zi-,*^sz-,*^rz-,*^x-}",
    "QS L-Front{*^e-,*^i-,*^y-,*^f-,*^v-,*^p-,*^b-,*^m-,*^w-,*^ww-}",
    "QS L-Central{*^schwa-,*^a-,*^t-,*^d-,*^s-,*^si-,*^z-,*^zi-,*^n-,*^r-,*^l-,*^t-,*^d-,*^sz-,*^rz-,*^cz-,*^drz-,*^c-,*^dz-,*^ci-,*^dzi-}",
    "QS L-Back{*^o-,*^u-,*^k-,*^g-,*^ki-,*^gi-,*^ng-,*^x-,*^gs-}",
    "QS L-Front_Vowel{*^e-,*^i-,*^y-}",
    "QS L-Central_Vowel{*^a-,*^schwa-}",
    "QS L-Back_Vowel{*^o-,*^u-}",
    "QS L-High_Vowel{*^i-,*^y-,*^u-}",
    "QS L-Medium_Vowel{*^e-,*^o-}",
    "QS L-Low_Vowel{*^a-}",
    "QS L-Rounded_Vowel{*^o-,*^u-}",
    "QS L-Unrounded_Vowel{*^a-,*^e-,*^i-,*^y-}",
    "QS L-IVowel{*^i-}",
    "QS L-EVowel{*^e-}",
    "QS L-AVowel{*^a-}",
    "QS L-OVowel{*^o-}",
    "QS L-UVowel{*^u-}",
    "QS L-YVowel                          {*^y-}",
    "QS LL-SCHWAvowel                     {*^schwa-}",
    "QS L-Unvoiced_Consonant{*^gs-,*^p-,*^t-,*^k-,*^ki-,*^f-,*^v-,*^s-,*^sz-,*^x-,*^c-,*^cz-,*^ci-}",
    "QS L-Voiced_Consonant{*^b-,*^d-,*^g-,*^gi-,*^v-,*^z-,*^zi-,*^rz-,*^dz-,*^drz-,*^dzi-,*^m-,*^n-,*^ni-,*^ng-,*^l-,*^r-,*^w-,*^ww-,*^j-,*^jj-}",
    "QS L-Front_Consonant{*^f-,*^v-,*^f-,*^p-,*^b-,*^m-,*^w-,*^ww-}",
    "QS L-Central_Consonant{*^t-,*^d-,*^s-,*^si-,*^z-,*^zi-,*^n-,*^r-,*^l-,*^t-,*^d-,*^sz-,*^rz-,*^cz-,*^drz-,*^c-,*^dz-,*^ci-,*^dzi-}",
    "QS L-Back_Consonant{*^gs-,*^k-,*^g-,*^ki-,*^gi-,*^ng-,*^x-}",
    "QS L-Fortis_Consonant{*^gs-,*^cz-,*^f-,*^k-,*^p-,*^s-,*^sz-,*^t-,*^ci-,*^c-,*^ki-}",
    "QS L-Lenis_Consonant{*^drz-,*^v-,*^g-,*^b-,*^rz-,*^z-,*^d-,*^dzi-,*^dz-,*^gi-,*^zi-}",
    "QS L-Neigther_F_or_L{*^m-,*^n-,*^ni-,*^ng-,*^l-,*^r-,*^w-,*^ww-,*^j-,*^jj-}",
    "QS L-Voiced_Stop{*^b-,*^d-,*^g-}",
    "QS L-Unvoiced_Stop{*^gs-,*^p-,*^t-,*^k-}",
    "QS L-Front_Stop{*^b-,*^p-}",
    "QS L-Central_Stop{*^d-,*^t-}",
    "QS L-Back_Stop{*^gs-,*^g-,*^k-}",
    "QS L-Voiced_Fricative{*^v-,*^z-,*^zi-,*^rz-}",
    "QS L-Unvoiced_Fricative{*^f-,*^s-,*^si-,*^sz-,*^x-}",
    "QS L-Front_Fricative{*^f-,*^v-}",
    "QS L-Affricate_Consonant{*^dz-,*^drz-,*^dzi-,*^c-,*^cz-,*^ci-}",
    "QS L-silences{*^pau-}",
    "QS L-schwa                                    {*^schwa-}",
    "QS L-a{*^a-}",
    "QS L-e{*^e-}",
    "QS L-i{*^i-}",
    "QS L-y{*^y-}",
    "QS L-o{*^o-}",
    "QS L-u{*^u-}",
    "QS L-p{*^p-}",
    "QS L-b{*^b-}",
    "QS L-t{*^t-}",
    "QS L-d{*^d-}",
    "QS L-k{*^k-}",
    "QS L-ki{*^ki-}",
    "QS L-g{*^g-}",
    "QS L-gi{*^gi-}",
    "QS L-f{*^f-}",
    "QS L-v{*^v-}",
    "QS L-s{*^s-}",
    "QS L-si{*^si-}",
    "QS L-z{*^z-}",
    "QS L-zi{*^zi-}",
    "QS L-sz{*^sz-}",
    "QS L-rz{*^rz-}",
    "QS L-x{*^x-}",
    "QS L-c{*^c-}",
    "QS L-dz{*^dz-}",
    "QS L-cz{*^cz-}",
    "QS L-drz{*^drz-}",
    "QS L-ci{*^ci-}",
    "QS L-dzi{*^dzi-}",
    "QS L-m{*^m-}",
    "QS L-n{*^n-}",
    "QS L-ni{*^ni-}",
    "QS L-ng{*^ng-}",
    "QS L-l{*^l-}",
    "QS L-r{*^r-}",
    "QS L-w{*^w-}",
    "QS L-ww{*^ww-}",
    "QS L-j{*^j-}",
    "QS L-jj{*^jj-}",
    "QS L-gs                                       {*^gs-}",
]
QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_C = [
    "QS C-Stop{*-p+*,*-b+*,*-t+*,*-d+*,*-k+*,*-g+*,*-gs+*}",
    "QS C-Nasal{*-ww+*,*-jj+*,*-m+*,*-n+*,*-ni+*,*-ng+*}",
    "QS C-Fricative{*-f+*,*-v+*,*-s+*,*-si+*,*-z+*,*-zi+*,*-sz+*,*-rz+*,*-x+*}",
    "QS C-Front{*-e+*,*-i+*,*-y+*,*-f+*,*-v+*,*-p+*,*-b+*,*-m+*,*-w+*,*-ww+*}",
    "QS C-Central{*-schwa+*,*-a+*,*-t+*,*-d+*,*-s+*,*-si+*,*-z+*,*-zi+*,*-n+*,*-r+*,*-l+*,*-t+*,*-d+*,*-sz+*,*-rz+*,*-cz+*,*-drz+*,*-c+*,*-dz+*,*-ci+*,*-dzi+*}",
    "QS C-Back{*-o+*,*-u+*,*-k+*,*-g+*,*-ki+*,*-gi+*,*-ng+*,*-x+*,*-gs+*}",
    "QS C-Front_Vowel{*-e+*,*-i+*,*-y+*}",
    "QS C-Central_Vowel{*-a+*,*-schwa+*}",
    "QS C-Back_Vowel{*-o+*,*-u+*}",
    "QS C-High_Vowel{*-i+*,*-y+*,*-u+*}",
    "QS C-Medium_Vowel{*-e+*,*-o+*}",
    "QS C-Low_Vowel{*-a+*}",
    "QS C-Rounded_Vowel{*-o+*,*-u+*}",
    "QS C-Unrounded_Vowel{*-a+*,*-e+*,*-i+*,*-y+*}",
    "QS C-IVowel{*-i+*}",
    "QS C-EVowel{*-e+*}",
    "QS C-AVowel{*-a+*}",
    "QS C-OVowel{*-o+*}",
    "QS C-UVowel{*-u+*}",
    "QS C-YVowel                          {*-y+*}",
    "QS C-SCHWAVowel                      {*-schwa+*}",
    "QS C-Unvoiced_Consonant{*-p+*,*-t+*,*-k+*,*-ki+*,*-f+*,*-v+*,*-s+*,*-sz+*,*-x+*,*-c+*,*-cz+*,*-ci+*,*-gs+*}",
    "QS C-Voiced_Consonant{*-b+*,*-d+*,*-g+*,*-gi+*,*-v+*,*-z+*,*-zi+*,*-rz+*,*-dz+*,*-drz+*,*-dzi+*,*-m+*,*-n+*,*-ni+*,*-ng+*,*-l+*,*-r+*,*-w+*,*-ww+*,*-j+*,*-jj+*}",
    "QS C-Front_Consonant{*-f+*,*-v+*,*-f+*,*-p+*,*-b+*,*-m+*,*-w+*,*-ww+*}",
    "QS C-Central_Consonant{*-t+*,*-d+*,*-s+*,*-si+*,*-z+*,*-zi+*,*-n+*,*-r+*,*-l+*,*-t+*,*-d+*,*-sz+*,*-rz+*,*-cz+*,*-drz+*,*-c+*,*-dz+*,*-ci+*,*-dzi+*}",
    "QS C-Back_Consonant{*-k+*,*-g+*,*-ki+*,*-gi+*,*-ng+*,*-x+*,*-gs+*}",
    "QS C-Fortis_Consonant{*-cz+*,*-f+*,*-k+*,*-p+*,*-s+*,*-sz+*,*-t+*,*-ci+*,*-c+*,*-ki+*,*-gs+*}",
    "QS C-Lenis_Consonant{*-drz+*,*-v+*,*-g+*,*-b+*,*-rz+*,*-z+*,*-d+*,*-dzi+*,*-dz+*,*-gi+*,*-zi+*}",
    "QS C-Neigther_F_or_L{*-m+*,*-n+*,*-ni+*,*-ng+*,*-l+*,*-r+*,*-w+*,*-ww+*,*-j+*,*-jj+*}",
    "QS C-Voiced_Stop{*-b+*,*-d+*,*-g+*}",
    "QS C-Unvoiced_Stop{*-p+*,*-t+*,*-k+*,*-gs+*}",
    "QS C-Front_Stop{*-b+*,*-p+*}",
    "QS C-Central_Stop{*-d+*,*-t+*}",
    "QS C-Back_Stop{*-g+*,*-k+*,*-gs+*}",
    "QS C-Voiced_Fricative{*-v+*,*-z+*,*-zi+*,*-rz+*}",
    "QS C-Unvoiced_Fricative{*-f+*,*-s+*,*-si+*,*-sz+*,*-x+*}",
    "QS C-Front_Fricative{*-f+*,*-v+*}",
    "QS C-Affricate_Consonant{*-dz+*,*-drz+*,*-dzi+*,*-c+*,*-cz+*,*-ci+*}",
    "QS C-silences{*-pau+*}",
    "QS C-schwa                                    {*-schwa+*}",
    "QS C-a{*-a+*}",
    "QS C-e{*-e+*}",
    "QS C-i{*-i+*}",
    "QS C-y{*-y+*}",
    "QS C-o{*-o+*}",
    "QS C-u{*-u+*}",
    "QS C-p{*-p+*}",
    "QS C-b{*-b+*}",
    "QS C-t{*-t+*}",
    "QS C-d{*-d+*}",
    "QS C-k{*-k+*}",
    "QS C-ki{*-ki+*}",
    "QS C-g{*-g+*}",
    "QS C-gi{*-gi+*}",
    "QS C-f{*-f+*}",
    "QS C-v{*-v+*}",
    "QS C-s{*-s+*}",
    "QS C-si{*-si+*}",
    "QS C-z{*-z+*}",
    "QS C-zi{*-zi+*}",
    "QS C-sz{*-sz+*}",
    "QS C-rz{*-rz+*}",
    "QS C-x{*-x+*}",
    "QS C-c{*-c+*}",
    "QS C-dz{*-dz+*}",
    "QS C-cz{*-cz+*}",
    "QS C-drz{*-drz+*}",
    "QS C-ci{*-ci+*}",
    "QS C-dzi{*-dzi+*}",
    "QS C-m{*-m+*}",
    "QS C-n{*-n+*}",
    "QS C-ni{*-ni+*}",
    "QS C-ng{*-ng+*}",
    "QS C-l{*-l+*}",
    "QS C-r{*-r+*}",
    "QS C-w{*-w+*}",
    "QS C-ww{*-ww+*}",
    "QS C-j{*-j+*}",
    "QS C-jj{*-jj+*}",
    "QS C-gs                                       {*-gs+*}",
]
QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_R = [
    "QS R-Stop{*+p=*,*+b=*,*+t=*,*+d=*,*+k=*,*+g=*,*+gs=*}",
    "QS R-Nasal{*+ww=*,*+jj=*,*+m=*,*+n=*,*+ni=*,*+ng=*}",
    "QS R-Fricative{*+f=*,*+v=*,*+s=*,*+si=*,*+z=*,*+zi=*,*+sz=*,*+rz=*,*+x=*}",
    "QS R-Front{*+e=*,*+i=*,*+y=*,*+f=*,*+v=*,*+p=*,*+b=*,*+m=*,*+w=*,*+ww=*}",
    "QS R-Central{*+schwa=*,*+a=*,*+t=*,*+d=*,*+s=*,*+si=*,*+z=*,*+zi=*,*+n=*,*+r=*,*+l=*,*+t=*,*+d=*,*+sz=*,*+rz=*,*+cz=*,*+drz=*,*+c=*,*+dz=*,*+ci=*,*+dzi=*}",
    "QS R-Back{*+o=*,*+u=*,*+k=*,*+g=*,*+ki=*,*+gi=*,*+ng=*,*+x=*,*+gs=*}",
    "QS R-Front_Vowel{*+e=*,*+i=*,*+y=*}",
    "QS R-Central_Vowel{*+a=*,*+schwa=*}",
    "QS R-Back_Vowel{*+o=*,*+u=*}",
    "QS R-High_Vowel{*+i=*,*+y=*,*+u=*}",
    "QS R-Medium_Vowel{*+e=*,*+o=*}",
    "QS R-Low_Vowel{*+a=*}",
    "QS R-Rounded_Vowel{*+o=*,*+u=*}",
    "QS R-Unrounded_Vowel{*+a=*,*+e=*,*+i=*,*+y=*}",
    "QS R-IVowel{*+i=*}",
    "QS R-OVowel{*+o=*}",
    "QS R-UVowel{*+u=*}",
    "QS R-YVowel                          {*+y=*}",
    "QS R-SCHWAVowel                      {*+schwa=*}",
    "QS R-Unvoiced_Consonant{*+p=*,*+t=*,*+k=*,*+ki=*,*+f=*,*+v=*,*+s=*,*+sz=*,*+x=*,*+c=*,*+cz=*,*+ci=*,*+gs=*}",
    "QS R-Voiced_Consonant{*+b=*,*+d=*,*+g=*,*+gi=*,*+v=*,*+z=*,*+zi=*,*+rz=*,*+dz=*,*+drz=*,*+dzi=*,*+m=*,*+n=*,*+ni=*,*+ng=*,*+l=*,*+r=*,*+w=*,*+ww=*,*+j=*,*+jj=*}",
    "QS R-Front_Consonant{*+f=*,*+v=*,*+f=*,*+p=*,*+b=*,*+m=*,*+w=*,*+ww=*}",
    "QS R-Central_Consonant{*+t=*,*+d=*,*+s=*,*+si=*,*+z=*,*+zi=*,*+n=*,*+r=*,*+l=*,*+t=*,*+d=*,*+sz=*,*+rz=*,*+cz=*,*+drz=*,*+c=*,*+dz=*,*+ci=*,*+dzi=*}",
    "QS R-Back_Consonant{*+k=*,*+g=*,*+ki=*,*+gi=*,*+ng=*,*+x=*,*+gs=*}",
    "QS R-Fortis_Consonant{*+cz=*,*+f=*,*+k=*,*+p=*,*+s=*,*+sz=*,*+t=*,*+ci=*,*+c=*,*+ki=*,*+gs=*}",
    "QS R-Lenis_Consonant{*+drz=*,*+v=*,*+g=*,*+b=*,*+rz=*,*+z=*,*+d=*,*+dzi=*,*+dz=*,*+gi=*,*+zi=*}",
    "QS R-Neigther_F_or_L{*+m=*,*+n=*,*+ni=*,*+ng=*,*+l=*,*+r=*,*+w=*,*+ww=*,*+j=*,*+jj=*}",
    "QS R-Voiced_Stop{*+b=*,*+d=*,*+g=*}",
    "QS R-Unvoiced_Stop{*+p=*,*+t=*,*+k=*,*+gs=*}",
    "QS R-Front_Stop{*+b=*,*+p=*}",
    "QS R-Central_Stop{*+d=*,*+t=*}",
    "QS R-Back_Stop{*+g=*,*+k=*,*+gs=*}",
    "QS R-Voiced_Fricative{*+v=*,*+z=*,*+zi=*,*+rz=*}",
    "QS R-Unvoiced_Fricative{*+f=*,*+s=*,*+si=*,*+sz=*,*+x=*}",
    "QS R-Front_Fricative{*+f=*,*+v=*}",
    "QS R-Affricate_Consonant{*+dz=*,*+drz=*,*+dzi=*,*+c=*,*+cz=*,*+ci=*}",
    "QS R-silences{*+pau=*}",
    "QS R-schwa                                    {*+schwa=*}",
    "QS R-a{*+a=*}",
    "QS R-e{*+e=*}",
    "QS R-i{*+i=*}",
    "QS R-y{*+y=*}",
    "QS R-o{*+o=*}",
    "QS R-u{*+u=*}",
    "QS R-p{*+p=*}",
    "QS R-b{*+b=*}",
    "QS R-t{*+t=*}",
    "QS R-d{*+d=*}",
    "QS R-k{*+k=*}",
    "QS R-ki{*+ki=*}",
    "QS R-g{*+g=*}",
    "QS R-gi{*+gi=*}",
    "QS R-f{*+f=*}",
    "QS R-v{*+v=*}",
    "QS R-s{*+s=*}",
    "QS R-si{*+si=*}",
    "QS R-z{*+z=*}",
    "QS R-zi{*+zi=*}",
    "QS R-sz{*+sz=*}",
    "QS R-rz{*+rz=*}",
    "QS R-x{*+x=*}",
    "QS R-c{*+c=*}",
    "QS R-dz{*+dz=*}",
    "QS R-cz{*+cz=*}",
    "QS R-drz{*+drz=*}",
    "QS R-ci{*+ci=*}",
    "QS R-dzi{*+dzi=*}",
    "QS R-m{*+m=*}",
    "QS R-n{*+n=*}",
    "QS R-ni{*+ni=*}",
    "QS R-ng{*+ng=*}",
    "QS R-l{*+l=*}",
    "QS R-r{*+r=*}",
    "QS R-w{*+w=*}",
    "QS R-ww{*+ww=*}",
    "QS R-j{*+j=*}",
    "QS R-jj{*+jj=*}",
    "QS R-gs{*+gs=*}",
]
QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_RR = [
    "QS RR-Stop{*=gs@*,*=p@*,*=b@*,*=t@*,*=d@*,*=k@*,*=g@*}",
    "QS RR-Nasal{*=ww@*,*=jj@*,*=m@*,*=n@*,*=ni@*,*=ng@*}",
    "QS RR-Fricative{*=f@*,*=v@*,*=s@*,*=si@*,*=z@*,*=zi@*,*=sz@*,*=rz@*,*=x@*}",
    "QS RR-Front{*=e@*,*=i@*,*=y@*,*=f@*,*=v@*,*=p@*,*=b@*,*=m@*,*=w@*,*=ww@*}",
    "QS RR-Central{*=schwa@*,*=a@*,*=t@*,*=d@*,*=s@*,*=si@*,*=z@*,*=zi@*,*=n@*,*=r@*,*=l@*,*=t@*,*=d@*,*=sz@*,*=rz@*,*=cz@*,*=drz@*,*=c@*,*=dz@*,*=ci@*,*=dzi@*}",
    "QS RR-Back{*=o@*,*=u@*,*=k@*,*=g@*,*=ki@*,*=gi@*,*=ng@*,*=x@*,*=gs@*}",
    "QS RR-Front_Vowel{*=e@*,*=i@*,*=y@*}",
    "QS RR-Central_Vowel{*=a@*,*=schwa@*}",
    "QS RR-Back_Vowel{*=o@*,*=u@*}",
    "QS RR-High_Vowel{*=i@*,*=y@*,*=u@*}",
    "QS RR-Medium_Vowel{*=e@*,*=o@*}",
    "QS RR-Low_Vowel{*=a@*}",
    "QS RR-Rounded_Vowel{*=o@*,*=u@*}",
    "QS RR-Unrounded_Vowel{*=a@*,*=e@*,*=i@*,*=y@*}",
    "QS RR-IVowel{*=i@*}",
    "QS RR-OVowel{*=o@*}",
    "QS RR-UVowel{*=u@*}",
    "QS RR-YVowel                          {*=y@*}",
    "QS RR-SCHWAVowel                      {*=schwa@*}",
    "QS RR-Unvoiced_Consonant{*=p@*,*=t@*,*=k@*,*=ki@*,*=f@*,*=v@*,*=s@*,*=sz@*,*=x@*,*=c@*,*=cz@*,*=ci@*,*=gs@*}",
    "QS RR-Voiced_Consonant{*=b@*,*=d@*,*=g@*,*=gi@*,*=v@*,*=z@*,*=zi@*,*=rz@*,*=dz@*,*=drz@*,*=dzi@*,*=m@*,*=n@*,*=ni@*,*=ng@*,*=l@*,*=r@*,*=w@*,*=ww@*,*=j@*,*=jj@*}",
    "QS RR-Front_Consonant{*=f@*,*=v@*,*=f@*,*=p@*,*=b@*,*=m@*,*=w@*,*=ww@*}",
    "QS RR-Central_Consonant{*=t@*,*=d@*,*=s@*,*=si@*,*=z@*,*=zi@*,*=n@*,*=r@*,*=l@*,*=t@*,*=d@*,*=sz@*,*=rz@*,*=cz@*,*=drz@*,*=c@*,*=dz@*,*=ci@*,*=dzi@*}",
    "QS RR-Back_Consonant{*=k@*,*=g@*,*=ki@*,*=gi@*,*=ng@*,*=x@*,*=gs@*}",
    "QS RR-Fortis_Consonant{*=cz@*,*=f@*,*=k@*,*=p@*,*=s@*,*=sz@*,*=t@*,*=ci@*,*=c@*,*=ki@*,*=gs@*}",
    "QS RR-Lenis_Consonant{*=drz@*,*=v@*,*=g@*,*=b@*,*=rz@*,*=z@*,*=d@*,*=dzi@*,*=dz@*,*=gi@*,*=zi@*}",
    "QS RR-Neigther_F_or_L{*=m@*,*=n@*,*=ni@*,*=ng@*,*=l@*,*=r@*,*=w@*,*=ww@*,*=j@*,*=jj@*}",
    "QS RR-Voiced_Stop{*=b@*,*=d@*,*=g@*}",
    "QS RR-Unvoiced_Stop{*=p@*,*=t@*,*=k@*,*=gs@*}",
    "QS RR-Front_Stop{*=b@*,*=p@*}",
    "QS RR-Central_Stop{*=d@*,*=t@*}",
    "QS RR-Back_Stop{*=g@*,*=k@*,*=gs@*}",
    "QS RR-Voiced_Fricative{*=v@*,*=z@*,*=zi@*,*=rz@*}",
    "QS RR-Unvoiced_Fricative{*=f@*,*=s@*,*=si@*,*=sz@*,*=x@*}",
    "QS RR-Front_Fricative{*=f@*,*=v@*}",
    "QS RR-Affricate_Consonant{*=dz@*,*=drz@*,*=dzi@*,*=c@*,*=cz@*,*=ci@*}",
    "QS RR-silences{*=pau@*}",
    "QS RR-a{*=a@*}",
    "QS RR-e{*=e@*}",
    "QS RR-i{*=i@*}",
    "QS RR-y{*=y@*}",
    "QS RR-o{*=o@*}",
    "QS RR-u{*=u@*}",
    "QS RR-p{*=p@*}",
    "QS RR-b{*=b@*}",
    "QS RR-t{*=t@*}",
    "QS RR-d{*=d@*}",
    "QS RR-k{*=k@*}",
    "QS RR-ki{*=ki@*}",
    "QS RR-g{*=g@*}",
    "QS RR-gi{*=gi@*}",
    "QS RR-f{*=f@*}",
    "QS RR-v{*=v@*}",
    "QS RR-s{*=s@*}",
    "QS RR-si{*=si@*}",
    "QS RR-z{*=z@*}",
    "QS RR-zi{*=zi@*}",
    "QS RR-sz{*=sz@*}",
    "QS RR-rz{*=rz@*}",
    "QS RR-x{*=x@*}",
    "QS RR-c{*=c@*}",
    "QS RR-dz{*=dz@*}",
    "QS RR-cz{*=cz@*}",
    "QS RR-drz{*=drz@*}",
    "QS RR-ci{*=ci@*}",
    "QS RR-dzi{*=dzi@*}",
    "QS RR-m{*=m@*}",
    "QS RR-n{*=n@*}",
    "QS RR-ni{*=ni@*}",
    "QS RR-ng{*=ng@*}",
    "QS RR-l{*=l@*}",
    "QS RR-r{*=r@*}",
    "QS RR-w{*=w@*}",
    "QS RR-ww{*=ww@*}",
    "QS RR-j{*=j@*}",
    "QS RR-jj{*=jj@*}",
    "QS RR-gs                                      {*=gs@*}",
]

NUM_SEG_IN_SYL_FW_EQ = [
    "QS Seg_Fw==x{*@x_*}",
    "QS Seg_Fw==1{*@1_*}",
    "QS Seg_Fw==2{*@2_*}",
    "QS Seg_Fw==3{*@3_*}",
    "QS Seg_Fw==4{*@4_*}",
    "QS Seg_Fw==5{*@5_*}",
    "QS Seg_Fw==6{*@6_*}",
    "QS Seg_Fw==7{*@7_*}",
]
NUM_SEG_IN_SYL_FW_LTE = [
    "QS Seg_Fw<=1{*@x_*,*@1_*}",
    "QS Seg_Fw<=2{*@x_*,*@1_*,*@2_*}",
    "QS Seg_Fw<=3{*@x_*,*@1_*,*@2_*,*@3_*}",
    "QS Seg_Fw<=4{*@x_*,*@1_*,*@2_*,*@3_*,*@4_*}",
    "QS Seg_Fw<=5{*@x_*,*@1_*,*@2_*,*@3_*,*@4_*,*@5_*}",
    "QS Seg_Fw<=6{*@x_*,*@1_*,*@2_*,*@3_*,*@4_*,*@5_*,*@6_*}",
    "QS Seg_Fw<=7{*@x_*,*@1_*,*@2_*,*@3_*,*@4_*,*@5_*,*@6_*,*@7_*}",
]
NUM_SEG_IN_SYL_BW_EQ = [
    "QS Seg_Bw==x{*_x/A:*}",
    "QS Seg_Bw==1{*_1/A:*}",
    "QS Seg_Bw==2{*_2/A:*}",
    "QS Seg_Bw==3{*_3/A:*}",
    "QS Seg_Bw==4{*_4/A:*}",
    "QS Seg_Bw==5{*_5/A:*}",
    "QS Seg_Bw==6{*_6/A:*}",
    "QS Seg_Bw==7{*_7/A:*}",
]
NUM_SEG_IN_SYL_BW_LTE = [
    "QS Seg_Bw<=0{*_x/A:*,*_0/A:*}",
    "QS Seg_Bw<=1{*_x/A:*,*_0/A:*,*_1/A:*}",
    "QS Seg_Bw<=2{*_x/A:*,*_0/A:*,*_1/A:*,*_2/A:*}",
    "QS Seg_Bw<=3{*_x/A:*,*_0/A:*,*_1/A:*,*_2/A:*,*_3/A:*}",
    "QS Seg_Bw<=4{*_x/A:*,*_0/A:*,*_1/A:*,*_2/A:*,*_3/A:*,*_4/A:*}",
    "QS Seg_Bw<=5{*_x/A:*,*_0/A:*,*_1/A:*,*_2/A:*,*_3/A:*,*_4/A:*,*_5/A:*}",
    "QS Seg_Bw<=6{*_x/A:*,*_0/A:*,*_1/A:*,*_2/A:*,*_3/A:*,*_4/A:*,*_5/A:*,*_6/A:*}",
    "QS Seg_Bw<=7{*_x/A:*,*_0/A:*,*_1/A:*,*_2/A:*,*_3/A:*,*_4/A:*,*_5/A:*,*_6/A:*,*_7/A:*}",
]

SYL_STRESSED_L = [
    "QS L-Syl_Stress==1{*/A:1_*}",
    "QS L-Syl_Stress==0{*/A:0_*}",
]
SYL_ACCENTED_L = [
    "QS L-Syl_Accent==1{*_1_*}",
    "QS L-Syl_Accent==0{*_0_*}",
]
SYL_ACCENT_TYPE_L = [
    "QS L-Syl_TOBI_Accent-H*{*/K:H?/L:*}",
    "QS L-Syl_TOBI_Accent-L*{*/K:L?/L:*}",
    "QS L-Syl_TOBI_Accent-L*+H{*/K:L?+H*}",
    "QS L-Syl_TOBI_Accent-L+H*{*/K:L+H*}",
    "QS L-Syl_TOBI_Accent-0{*/K:0*}",
    "QS L-Syl_TOBI_Accent-NONE{*/K:NONE*}",
    "QS L-Syl_TOBI_Accent-x{*/K:x*}",
]

SYL_LEN_IN_NO_SEG_EQ_L = [
    "QS L-Syl_Num-Segs==0{*_0/B:*}",
    "QS L-Syl_Num-Segs==1{*_1/B:*}",
    "QS L-Syl_Num-Segs==2{*_2/B:*}",
    "QS L-Syl_Num-Segs==3{*_3/B:*}",
    "QS L-Syl_Num-Segs==4{*_4/B:*}",
    "QS L-Syl_Num-Segs==5{*_5/B:*}",
    "QS L-Syl_Num-Segs==6{*_6/B:*}",
    "QS L-Syl_Num-Segs==7{*_7/B:*}",
]
SYL_LEN_IN_NO_SEG_LTE_L =[
    "QS L-Syl_Num-Segs<=1{*_0/B:*,*_1/B:*}",
    "QS L-Syl_Num-Segs<=2{*_0/B:*,*_1/B:*,*_2/B:*}",
    "QS L-Syl_Num-Segs<=3{*_0/B:*,*_1/B:*,*_2/B:*,*_3/B:*}",
    "QS L-Syl_Num-Segs<=4{*_0/B:*,*_1/B:*,*_2/B:*,*_3/B:*,*_4/B:*}",
    "QS L-Syl_Num-Segs<=5{*_0/B:*,*_1/B:*,*_2/B:*,*_3/B:*,*_4/B:*,*_5/B:*}",
    "QS L-Syl_Num-Segs<=6{*_0/B:*,*_1/B:*,*_2/B:*,*_3/B:*,*_4/B:*,*_5/B:*,*_6/B:*}",
    "QS L-Syl_Num-Segs<=7{*_0/B:*,*_1/B:*,*_2/B:*,*_3/B:*,*_4/B:*,*_5/B:*,*_6/B:*,*_7/B:*}",
]

SYL_STRESSED_C = [
    "QS C-Syl_Stress==1{*/B:1-*}",
    "QS C-Syl_Stress==0{*/B:0-*}",
    "QS C-Syl_Stress==x{*/B:x-*}",
]
SYL_ACCENTED_C = [
    "QS C-Syl_Accent==1{*-1-*}",
    "QS C-Syl_Accent==0{*-0-*}",
    "QS C-Syl_Accent==x{*-x-*}",
]
SYL_ACCENT_TYPE_C = [
    "QS C-Syl_TOBI_Accent-H*{*/L:H?/M:*}",
    "QS C-Syl_TOBI_Accent-L*{*/L:L?/M:*}",
    "QS C-Syl_TOBI_Accent-L*+H{*/L:L?+H*}",
    "QS C-Syl_TOBI_Accent-L+H*{*/L:L+H*}",
    "QS C-Syl_TOBI_Accent-0{*/L:0*}",
    "QS C-Syl_TOBI_Accent-NONE{*/L:NONE*}",
    "QS C-Syl_TOBI_Accent-x{*/L:x*}",
]

SYL_LEN_IN_NO_SEG_EQ_C = [
    "QS C-Syl_Num-Segs==x{*-x@*}",
    "QS C-Syl_Num-Segs==1{*-1@*}",
    "QS C-Syl_Num-Segs==2{*-2@*}",
    "QS C-Syl_Num-Segs==3{*-3@*}",
    "QS C-Syl_Num-Segs==4{*-4@*}",
    "QS C-Syl_Num-Segs==5{*-5@*}",
    "QS C-Syl_Num-Segs==6{*-6@*}",
    "QS C-Syl_Num-Segs==7{*-7@*}",
]
SYL_LEN_IN_NO_SEG_LTE_C = [
    "QS C-Syl_Num-Segs<=1{*-x@*,*-1@*}",
    "QS C-Syl_Num-Segs<=2{*-x@*,*-1@*,*-2@*}",
    "QS C-Syl_Num-Segs<=3{*-x@*,*-1@*,*-2@*,*-3@*}",
    "QS C-Syl_Num-Segs<=4{*-x@*,*-1@*,*-2@*,*-3@*,*-4@*}",
    "QS C-Syl_Num-Segs<=5{*-x@*,*-1@*,*-2@*,*-3@*,*-4@*,*-5@*}",
    "QS C-Syl_Num-Segs<=6{*-x@*,*-1@*,*-2@*,*-3@*,*-4@*,*-5@*,*-6@*}",
    "QS C-Syl_Num-Segs<=7{*-x@*,*-1@*,*-2@*,*-3@*,*-4@*,*-5@*,*-6@*,*-7@*}",
]

SYL_POSITION_IN_WORD_FW_EQ = [
    "QS Pos_C-Syl_in_C-Word(Fw)==x{*@x-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==1{*@1-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==2{*@2-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==3{*@3-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==4{*@4-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==5{*@5-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==6{*@6-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)==7{*@7-*}",
]
SYL_POSITION_IN_WORD_FW_LTE = [
    "QS Pos_C-Syl_in_C-Word(Fw)<=1{*@x-*,*@1-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)<=2{*@x-*,*@1-*,*@2-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)<=3{*@x-*,*@1-*,*@2-*,*@3-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)<=4{*@x-*,*@1-*,*@2-*,*@3-*,*@4-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)<=5{*@x-*,*@1-*,*@2-*,*@3-*,*@4-*,*@5-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)<=6{*@x-*,*@1-*,*@2-*,*@3-*,*@4-*,*@5-*,*@6-*}",
    "QS Pos_C-Syl_in_C-Word(Fw)<=7{*@x-*,*@1-*,*@2-*,*@3-*,*@4-*,*@5-*,*@6-*,*@7-*}",
]
SYL_POSITION_IN_WORD_BW_EQ = [
    "QS Pos_C-Syl_in_C-Word(Bw)==x{*-x&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==1{*-1&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==2{*-2&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==3{*-3&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==4{*-4&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==5{*-5&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==6{*-6&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)==7{*-7&*}",
]
SYL_POSITION_IN_WORD_BW_LTE = [
    "QS Pos_C-Syl_in_C-Word(Bw)<=1{*-x&*,*-1&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)<=2{*-x&*,*-1&*,*-2&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)<=3{*-x&*,*-1&*,*-2&*,*-3&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)<=4{*-x&*,*-1&*,*-2&*,*-3&*,*-4&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)<=5{*-x&*,*-1&*,*-2&*,*-3&*,*-4&*,*-5&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)<=6{*-x&*,*-1&*,*-2&*,*-3&*,*-4&*,*-5&*,*-6&*}",
    "QS Pos_C-Syl_in_C-Word(Bw)<=7{*-x&*,*-1&*,*-2&*,*-3&*,*-4&*,*-5&*,*-6&*,*-7&*}",
]
SYL_POSITION_IN_PHRASE_FW_EQ = [
    "QS Pos_C-Syl_in_C-Phrase(Fw)==x{*&x-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==1{*&1-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==2{*&2-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==3{*&3-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==4{*&4-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==5{*&5-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==6{*&6-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==7{*&7-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==8{*&8-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==9{*&9-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==10{*&10-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==11{*&11-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==12{*&12-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==13{*&13-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==14{*&14-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==15{*&15-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==16{*&16-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==17{*&17-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==18{*&18-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==19{*&19-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)==20{*&20-*}",
]
SYL_POSITION_IN_PHRASE_FW_LTE = [
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=1{*&x-*,*&0-*,*&1-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=2{*&x-*,*&0-*,*&1-*,*&2-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=3{*&x-*,*&0-*,*&1-*,*&2-*,*&3-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=4{*&x-*,*&0-*,*&1-*,*&2-*,*&3-*,*&4-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=5{*&x-*,*&0-*,*&1-*,*&2-*,*&3-*,*&4-*,*&5-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=6{*&x-*,*&0-*,*&1-*,*&2-*,*&3-*,*&4-*,*&5-*,*&6-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=7{*&x-*,*&0-*,*&1-*,*&2-*,*&3-*,*&4-*,*&5-*,*&6-*,*&7-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=8{*&x-*,*&0-*,*&1-*,*&2-*,*&3-*,*&4-*,*&5-*,*&6-*,*&7-*,*&8-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=9{*&?-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=10{*&?-*,*&10-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=11{*&?-*,*&10-*,*&11-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=12{*&?-*,*&10-*,*&11-*,*&12-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=13{*&?-*,*&10-*,*&11-*,*&12-*,*&13-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=14{*&?-*,*&10-*,*&11-*,*&12-*,*&13-*,*&14-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=15{*&?-*,*&10-*,*&11-*,*&12-*,*&13-*,*&14-*,*&15-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=16{*&?-*,*&10-*,*&11-*,*&12-*,*&13-*,*&14-*,*&15-*,*&16-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=17{*&?-*,*&10-*,*&11-*,*&12-*,*&13-*,*&14-*,*&15-*,*&16-*,*&17-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=18{*&?-*,*&10-*,*&11-*,*&12-*,*&13-*,*&14-*,*&15-*,*&16-*,*&17-*,*&18-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=19{*&?-*,*&1?-*}",
    "QS Pos_C-Syl_in_C-Phrase(Fw)<=20{*&?-*,*&1?-*,*&20-*}",
]
SYL_POSITION_IN_PHRASE_BW_EQ = [
    "QS Pos_C-Syl_in_C-Phrase(Bw)==x{*-x#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==1{*-1#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==2{*-2#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==3{*-3#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==4{*-4#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==5{*-5#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==6{*-6#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==7{*-7#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==8{*-8#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==9{*-9#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==10{*-10#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==11{*-11#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==12{*-12#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==13{*-13#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==14{*-14#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==15{*-15#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==16{*-16#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==17{*-17#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==18{*-18#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==19{*-19#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)==20{*-20#*}",
]
SYL_POSITION_IN_PHRASE_BW_LTE = [
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=1{*-x#*,*-1#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=2{*-x#*,*-1#*,*-2#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=3{*-x#*,*-1#*,*-2#*,*-3#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=4{*-x#*,*-1#*,*-2#*,*-3#*,*-4#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=5{*-x#*,*-1#*,*-2#*,*-3#*,*-4#*,*-5#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=6{*-x#*,*-1#*,*-2#*,*-3#*,*-4#*,*-5#*,*-6#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=7{*-x#*,*-1#*,*-2#*,*-3#*,*-4#*,*-5#*,*-6#*,*-7#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=8{*-x#*,*-1#*,*-2#*,*-3#*,*-4#*,*-5#*,*-6#*,*-7#*,*-8#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=9{*-?#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=10{*-?#*,*-10#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=11{*-?#*,*-10#*,*-11#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=12{*-?#*,*-10#*,*-11#*,*-12#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=13{*-?#*,*-10#*,*-11#*,*-12#*,*-13#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=14{*-?#*,*-10#*,*-11#*,*-12#*,*-13#*,*-14#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=15{*-?#*,*-10#*,*-11#*,*-12#*,*-13#*,*-14#*,*-15#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=16{*-?#*,*-10#*,*-11#*,*-12#*,*-13#*,*-14#*,*-15#*,*-16#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=17{*-?#*,*-10#*,*-11#*,*-12#*,*-13#*,*-14#*,*-15#*,*-16#*,*-17#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=18{*-?#*,*-10#*,*-11#*,*-12#*,*-13#*,*-14#*,*-15#*,*-16#*,*-17#*,*-18#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=19{*-?#*,*-1?#*}",
    "QS Pos_C-Syl_in_C-Phrase(Bw)<=20{*-?#*,*-1?#*,*-20#*}",
]

NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ = [
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==x{*#x-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==1{*#1-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==2{*#2-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==3{*#3-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==4{*#4-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==5{*#5-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==6{*#6-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==7{*#7-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==8{*#8-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==9{*#9-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==10{*#10-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==11{*#11-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase==12{*#12-*}",
]
NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE = [
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=1{*#x-*,*#1-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=2{*#x-*,*#1-*,*#2-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=3{*#x-*,*#1-*,*#2-*,*#3-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=4{*#x-*,*#1-*,*#2-*,*#3-*,*#4-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=5{*#x-*,*#1-*,*#2-*,*#3-*,*#4-*,*#5-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=6{*#x-*,*#1-*,*#2-*,*#3-*,*#4-*,*#5-*,*#6-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=7{*#x-*,*#1-*,*#2-*,*#3-*,*#4-*,*#5-*,*#6-*,*#7-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=8{*#x-*,*#1-*,*#2-*,*#3-*,*#4-*,*#5-*,*#6-*,*#7-*,*#8-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=9{*#?-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=10{*#?-*,*#10-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=11{*#?-*,*#10-*,*#11-*}",
    "QS Num-StressedSyl_before_C-Syl_in_C-Phrase<=12{*#?-*,*#10-*,*#11-*,*#12-*}",
]
NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ = [
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==x{*-x$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==1{*-1$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==2{*-2$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==3{*-3$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==4{*-4$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==5{*-5$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==6{*-6$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==7{*-7$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==8{*-8$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==9{*-9$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==10{*-10$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==11{*-11$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase==12{*-12$*}",
]
NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE = [
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=1{*-x$*,*-1$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=2{*-x$*,*-1$*,*-2$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=3{*-x$*,*-1$*,*-2$*,*-3$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=4{*-x$*,*-1$*,*-2$*,*-3$*,*-4$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=5{*-x$*,*-1$*,*-2$*,*-3$*,*-4$*,*-5$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=6{*-x$*,*-1$*,*-2$*,*-3$*,*-4$*,*-5$*,*-6$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=7{*-x$*,*-1$*,*-2$*,*-3$*,*-4$*,*-5$*,*-6$*,*-7$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=8{*-x$*,*-1$*,*-2$*,*-3$*,*-4$*,*-5$*,*-6$*,*-7$*,*-8$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=9{*-?$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=10{*-?$*,*-10$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=11{*-?$*,*-10$*,*-11$*}",
    "QS Num-StressedSyl_after_C-Syl_in_C-Phrase<=12{*-?$*,*-10$*,*-11$*,*-12$*}",
]
NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ = [
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==x{*$x-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==1{*$1-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==2{*$2-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==3{*$3-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==4{*$4-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==5{*$5-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase==6{*$6-*}",
]
NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE = [
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase<=1{*$x-*,*$1-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase<=2{*$x-*,*$1-*,*$2-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase<=3{*$x-*,*$1-*,*$2-*,*$3-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase<=4{*$x-*,*$1-*,*$2-*,*$3-*,*$4-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase<=5{*$x-*,*$1-*,*$2-*,*$3-*,*$4-*,*$5-*}",
    "QS Num-AccentedSyl_before_C-Syl_in_C-Phrase<=6{*$x-*,*$1-*,*$2-*,*$3-*,*$4-*,*$5-*,*$6-*}",
]
NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ = [
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==x{*-x!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==1{*-1!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==2{*-2!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==3{*-3!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==4{*-4!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==5{*-5!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==6{*-6!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase==7{*-7!*}",
]
NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE = [
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=1{*-x!*,*-1!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=2{*-x!*,*-1!*,*-2!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=3{*-x!*,*-1!*,*-2!*,*-3!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=4{*-x!*,*-1!*,*-2!*,*-3!*,*-4!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=5{*-x!*,*-1!*,*-2!*,*-3!*,*-4!*,*-5!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=6{*-x!*,*-1!*,*-2!*,*-3!*,*-4!*,*-5!*,*-6!*}",
    "QS Num-AccentedSyl_after_C-Syl_in_C-Phrase<=7{*-x!*,*-1!*,*-2!*,*-3!*,*-4!*,*-5!*,*-6!*,*-7!*}",
]
NUM_SYLS_FROM_PREV_STRESSED_SYL_EQ = [
    "QS Num-Syl_from_prev-StressedSyl==x{*!x-*}",
    "QS Num-Syl_from_prev-StressedSyl==0{*!0-*}",
    "QS Num-Syl_from_prev-StressedSyl==1{*!1-*}",
    "QS Num-Syl_from_prev-StressedSyl==2{*!2-*}",
    "QS Num-Syl_from_prev-StressedSyl==3{*!3-*}",
    "QS Num-Syl_from_prev-StressedSyl==4{*!4-*}",
    "QS Num-Syl_from_prev-StressedSyl==5{*!5-*}",
]
NUM_SYLS_FROM_PREV_STRESSED_SYL_LTE = [
    "QS Num-Syl_from_prev-StressedSyl<=0{*!x-*,*!0-*}",
    "QS Num-Syl_from_prev-StressedSyl<=1{*!x-*,*!0-*,*!1-*}",
    "QS Num-Syl_from_prev-StressedSyl<=2{*!x-*,*!0-*,*!1-*,*!2-*}",
    "QS Num-Syl_from_prev-StressedSyl<=3{*!x-*,*!0-*,*!1-*,*!2-*,*!3-*}",
    "QS Num-Syl_from_prev-StressedSyl<=4{*!x-*,*!0-*,*!1-*,*!2-*,*!3-*,*!4-*}",
    "QS Num-Syl_from_prev-StressedSyl<=5{*!x-*,*!0-*,*!1-*,*!2-*,*!3-*,*!4-*,*!5-*}",
]
NUM_SYLS_FROM_NEXT_STRESSED_SYL_EQ = [
    "QS Num-Syl_from_next-StressedSyl==x{*-x;*}",
    "QS Num-Syl_from_next-StressedSyl==0{*-0;*}",
    "QS Num-Syl_from_next-StressedSyl==1{*-1;*}",
    "QS Num-Syl_from_next-StressedSyl==2{*-2;*}",
    "QS Num-Syl_from_next-StressedSyl==3{*-3;*}",
    "QS Num-Syl_from_next-StressedSyl==4{*-4;*}",
    "QS Num-Syl_from_next-StressedSyl==5{*-5;*}",
]
NUM_SYLS_FROM_NEXT_STRESSED_SYL_LTE = [
    "QS Num-Syl_from_next-StressedSyl<=0{*-x;*,*-0;*}",
    "QS Num-Syl_from_next-StressedSyl<=1{*-x;*,*-0;*,*-1;*}",
    "QS Num-Syl_from_next-StressedSyl<=2{*-x;*,*-0;*,*-1;*,*-2;*}",
    "QS Num-Syl_from_next-StressedSyl<=3{*-x;*,*-0;*,*-1;*,*-2;*,*-3;*}",
    "QS Num-Syl_from_next-StressedSyl<=4{*-x;*,*-0;*,*-1;*,*-2;*,*-3;*,*-4;*}",
    "QS Num-Syl_from_next-StressedSyl<=5{*-x;*,*-0;*,*-1;*,*-2;*,*-3;*,*-4;*,*-5;*}",
]
NUM_SYLS_FROM_PREV_ACCENTED_SYL_EQ = [
    "QS Num-Syl_from_prev-AccentedSyl==x{*;x-*}",
    "QS Num-Syl_from_prev-AccentedSyl==0{*;0-*}",
    "QS Num-Syl_from_prev-AccentedSyl==1{*;1-*}",
    "QS Num-Syl_from_prev-AccentedSyl==2{*;2-*}",
    "QS Num-Syl_from_prev-AccentedSyl==3{*;3-*}",
    "QS Num-Syl_from_prev-AccentedSyl==4{*;4-*}",
    "QS Num-Syl_from_prev-AccentedSyl==5{*;5-*}",
    "QS Num-Syl_from_prev-AccentedSyl==6{*;6-*}",
    "QS Num-Syl_from_prev-AccentedSyl==7{*;7-*}",
    "QS Num-Syl_from_prev-AccentedSyl==8{*;8-*}",
    "QS Num-Syl_from_prev-AccentedSyl==9{*;9-*}",
    "QS Num-Syl_from_prev-AccentedSyl==10{*;10-*}",
    "QS Num-Syl_from_prev-AccentedSyl==11{*;11-*}",
    "QS Num-Syl_from_prev-AccentedSyl==12{*;12-*}",
    "QS Num-Syl_from_prev-AccentedSyl==13{*;13-*}",
    "QS Num-Syl_from_prev-AccentedSyl==14{*;14-*}",
    "QS Num-Syl_from_prev-AccentedSyl==15{*;15-*}",
    "QS Num-Syl_from_prev-AccentedSyl==16{*;16-*}",
]
NUM_SYLS_FROM_PREV_ACCENTED_SYL_LTE = [
    "QS Num-Syl_from_prev-AccentedSyl<=0{*;x-*,*;0-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=1{*;x-*,*;0-*,*;1-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=2{*;x-*,*;0-*,*;1-*,*;2-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=3{*;x-*,*;0-*,*;1-*,*;2-*,*;3-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=4{*;x-*,*;0-*,*;1-*,*;2-*,*;3-*,*;4-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=5{*;x-*,*;0-*,*;1-*,*;2-*,*;3-*,*;4-*,*;5-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=6{*;x-*,*;0-*,*;1-*,*;2-*,*;3-*,*;4-*,*;5-*,*;6-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=7{*;x-*,*;0-*,*;1-*,*;2-*,*;3-*,*;4-*,*;5-*,*;6-*,*;7-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=8{*;x-*,*;0-*,*;1-*,*;2-*,*;3-*,*;4-*,*;5-*,*;6-*,*;7-*,*;8-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=9{*;?-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=10{*;?-*,*;10-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=11{*;?-*,*;10-*,*;11-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=12{*;?-*,*;10-*,*;11-*,*;12-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=13{*;?-*,*;10-*,*;11-*,*;12-*,*;13-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=14{*;?-*,*;10-*,*;11-*,*;12-*,*;13-*,*;14-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=15{*;?-*,*;10-*,*;11-*,*;12-*,*;13-*,*;14-*,*;15-*}",
    "QS Num-Syl_from_prev-AccentedSyl<=16{*;?-*,*;10-*,*;11-*,*;12-*,*;13-*,*;14-*,*;15-*,*;16-*}",
]
NUM_SYLS_FROM_NEXT_ACCENTED_SYL_EQ = [
    "QS Num-Syl_from_next-AccentedSyl==x{*-x|*}",
    "QS Num-Syl_from_next-AccentedSyl==0{*-0|*}",
    "QS Num-Syl_from_next-AccentedSyl==1{*-1|*}",
    "QS Num-Syl_from_next-AccentedSyl==2{*-2|*}",
    "QS Num-Syl_from_next-AccentedSyl==3{*-3|*}",
    "QS Num-Syl_from_next-AccentedSyl==4{*-4|*}",
    "QS Num-Syl_from_next-AccentedSyl==5{*-5|*}",
    "QS Num-Syl_from_next-AccentedSyl==6{*-6|*}",
    "QS Num-Syl_from_next-AccentedSyl==7{*-7|*}",
    "QS Num-Syl_from_next-AccentedSyl==8{*-8|*}",
    "QS Num-Syl_from_next-AccentedSyl==9{*-9|*}",
    "QS Num-Syl_from_next-AccentedSyl==10{*-10|*}",
    "QS Num-Syl_from_next-AccentedSyl==11{*-11|*}",
    "QS Num-Syl_from_next-AccentedSyl==12{*-12|*}",
    "QS Num-Syl_from_next-AccentedSyl==13{*-13|*}",
    "QS Num-Syl_from_next-AccentedSyl==14{*-14|*}",
    "QS Num-Syl_from_next-AccentedSyl==15{*-15|*}",
    "QS Num-Syl_from_next-AccentedSyl==16{*-16|*}",
]
NUM_SYLS_FROM_NEXT_ACCENTED_SYL_LTE = [
    "QS Num-Syl_from_next-AccentedSyl<=0{*-x|*,*-0|*}",
    "QS Num-Syl_from_next-AccentedSyl<=1{*-x|*,*-0|*,*-1|*}",
    "QS Num-Syl_from_next-AccentedSyl<=2{*-x|*,*-0|*,*-1|*,*-2|*}",
    "QS Num-Syl_from_next-AccentedSyl<=3{*-x|*,*-0|*,*-1|*,*-2|*,*-3|*}",
    "QS Num-Syl_from_next-AccentedSyl<=4{*-x|*,*-0|*,*-1|*,*-2|*,*-3|*,*-4|*}",
    "QS Num-Syl_from_next-AccentedSyl<=5{*-x|*,*-0|*,*-1|*,*-2|*,*-3|*,*-4|*,*-5|*}",
    "QS Num-Syl_from_next-AccentedSyl<=6{*-x|*,*-0|*,*-1|*,*-2|*,*-3|*,*-4|*,*-5|*,*-6|*}",
    "QS Num-Syl_from_next-AccentedSyl<=7{*-x|*,*-0|*,*-1|*,*-2|*,*-3|*,*-4|*,*-5|*,*-6|*,*-7|*}",
    "QS Num-Syl_from_next-AccentedSyl<=8{*-x|*,*-0|*,*-1|*,*-2|*,*-3|*,*-4|*,*-5|*,*-6|*,*-7|*,*-8|*}",
    "QS Num-Syl_from_next-AccentedSyl<=9{*-?|*}",
    "QS Num-Syl_from_next-AccentedSyl<=10{*-?|*,*-10|*}",
    "QS Num-Syl_from_next-AccentedSyl<=11{*-?|*,*-10|*,*-11|*}",
    "QS Num-Syl_from_next-AccentedSyl<=12{*-?|*,*-10|*,*-11|*,*-12|*}",
    "QS Num-Syl_from_next-AccentedSyl<=13{*-?|*,*-10|*,*-11|*,*-12|*,*-13|*}",
    "QS Num-Syl_from_next-AccentedSyl<=14{*-?|*,*-10|*,*-11|*,*-12|*,*-13|*,*-14|*}",
    "QS Num-Syl_from_next-AccentedSyl<=15{*-?|*,*-10|*,*-11|*,*-12|*,*-13|*,*-14|*,*-15|*}",
    "QS Num-Syl_from_next-AccentedSyl<=16{*-?|*,*-10|*,*-11|*,*-12|*,*-13|*,*-14|*,*-15|*,*-16|*}",
]

SYL_VOWEL_TYPE = [
    "QS C-Syl_Vowel==x{*|x/C:*}",
    "QS C-Syl_Vowel{*|i/C:*,*|y/C:*,*|e/C:*,*|a/C:*,*|o/C:*,*|u/C:*,*|schwa/C:*}",
    "QS C-Syl_Front_Vowel{*|e/C:*,*|i/C:*,*|y/C:*}",
    "QS C-Syl_Central_Vowel{*|a/C:*,*|schwa/C:*}",
    "QS C-Syl_Back_Vowel{*|o/C:*,*|u/C:*}",
    "QS C-Syl_High_Vowel{*|i/C:*,*|y/C:*,*|u/C:*}",
    "QS C-Syl_Medium_Vowel{*|e/C:*,*|o/C:*,*|schwa/C:*}",
    "QS C-Syl_Low_Vowel{*|a/C:*}",
    "QS C-Syl_Rounded_Vowel{*|o/C:*,*|u/C:*}",
    "QS C-Syl_Unrounded_Vowel{*|a/C:*,*|e/C:*,*|i/C:*,*|y/C:*}",
    "QS C-Syl_IVowel{*|i/C:*}",
    "QS C-Syl_EVowel{*|e/C:*}",
    "QS C-Syl_AVowel{*|a/C:*}",
    "QS C-Syl_OVowel{*|o/C:*}",
    "QS C-Syl_UVowel{*|u/C:*}",
    "QS C-Syl_YVowel{*|y/C:*}",
    "QS C-Syl_SCHWAVowel                   {*|schwa/C:*}",
]

SYL_STRESSED_R = [
    "QS R-Syl_Stress==1{*/C:1+*}",
    "QS R-Syl_Stress==0{*/C:0+*}",
]
SYL_ACCENTED_R = [
    "QS R-Syl_Accent==1{*+1+*}",
    "QS R-Syl_Accent==0{*+0+*}",
]
SYL_ACCENT_TYPE_R = [
    "QS R-Syl_TOBI_Accent-H*{*/M:H?}",
    "QS R-Syl_TOBI_Accent-L*{*/M:L?}",
    "QS R-Syl_TOBI_Accent-L*+H{*/M:L?+H}",
    "QS R-Syl_TOBI_Accent-L+H*{*/M:L+H?}",
    "QS R-Syl_TOBI_Accent-0{*/M:0}",
    "QS R-Syl_TOBI_Accent-NONE{*/M:NONE}",
    "QS R-Syl_TOBI_Accent-x{*/M:x}",
]

SYL_LEN_IN_NO_SEG_EQ_R = [
    "QS R-Syl_Num-Segs==0{*+0/D:*}",
    "QS R-Syl_Num-Segs==1{*+1/D:*}",
    "QS R-Syl_Num-Segs==2{*+2/D:*}",
    "QS R-Syl_Num-Segs==3{*+3/D:*}",
    "QS R-Syl_Num-Segs==4{*+4/D:*}",
    "QS R-Syl_Num-Segs==5{*+5/D:*}",
    "QS R-Syl_Num-Segs==6{*+6/D:*}",
    "QS R-Syl_Num-Segs==7{*+7/D:*}",
]
SYL_LEN_IN_NO_SEG_LTE_R = [
    "QS R-Syl_Num-Segs<=1{*+0/D:*,*+1/D:*}",
    "QS R-Syl_Num-Segs<=2{*+0/D:*,*+1/D:*,*+2/D:*}",
    "QS R-Syl_Num-Segs<=3{*+0/D:*,*+1/D:*,*+2/D:*,*+3/D:*}",
    "QS R-Syl_Num-Segs<=4{*+0/D:*,*+1/D:*,*+2/D:*,*+3/D:*,*+4/D:*}",
    "QS R-Syl_Num-Segs<=5{*+0/D:*,*+1/D:*,*+2/D:*,*+3/D:*,*+4/D:*,*+5/D:*}",
    "QS R-Syl_Num-Segs<=6{*+0/D:*,*+1/D:*,*+2/D:*,*+3/D:*,*+4/D:*,*+5/D:*,*+6/D:*}",
    "QS R-Syl_Num-Segs<=7{*+0/D:*,*+1/D:*,*+2/D:*,*+3/D:*,*+4/D:*,*+5/D:*,*+6/D:*,*+7/D:*}",
]

WORD_LEN_IN_NO_SYLS_EQ_L = [
    "QS L-Word_Num-Syls==0{*_0/E:*}",
    "QS L-Word_Num-Syls==1{*_1/E:*}",
    "QS L-Word_Num-Syls==2{*_2/E:*}",
    "QS L-Word_Num-Syls==3{*_3/E:*}",
    "QS L-Word_Num-Syls==4{*_4/E:*}",
    "QS L-Word_Num-Syls==5{*_5/E:*}",
    "QS L-Word_Num-Syls==6{*_6/E:*}",
    "QS L-Word_Num-Syls==7{*_7/E:*}",
]
WORD_LEN_IN_NO_SYLS_LTE_L = [
    "QS L-Word_Num-Syls<=1{*_0/E:*,*_1/E:*}",
    "QS L-Word_Num-Syls<=2{*_0/E:*,*_1/E:*,*_2/E:*}",
    "QS L-Word_Num-Syls<=3{*_0/E:*,*_1/E:*,*_2/E:*,*_3/E:*}",
    "QS L-Word_Num-Syls<=4{*_0/E:*,*_1/E:*,*_2/E:*,*_3/E:*,*_4/E:*}",
    "QS L-Word_Num-Syls<=5{*_0/E:*,*_1/E:*,*_2/E:*,*_3/E:*,*_4/E:*,*_5/E:*}",
    "QS L-Word_Num-Syls<=6{*_0/E:*,*_1/E:*,*_2/E:*,*_3/E:*,*_4/E:*,*_5/E:*,*_6/E:*}",
    "QS L-Word_Num-Syls<=7{*_0/E:*,*_1/E:*,*_2/E:*,*_3/E:*,*_4/E:*,*_5/E:*,*_6/E:*,*_7/E:*}",
]
WORD_LEN_IN_NO_SYLS_EQ_C = [
    "QS C-Word_Num-Syls==x{*+x@*}",
    "QS C-Word_Num-Syls==1{*+1@*}",
    "QS C-Word_Num-Syls==2{*+2@*}",
    "QS C-Word_Num-Syls==3{*+3@*}",
    "QS C-Word_Num-Syls==4{*+4@*}",
    "QS C-Word_Num-Syls==5{*+5@*}",
    "QS C-Word_Num-Syls==6{*+6@*}",
    "QS C-Word_Num-Syls==7{*+7@*}",
]
WORD_LEN_IN_NO_SYLS_LTE_C = [
    "QS C-Word_Num-Syls<=1{*+x@*,*+1@*}",
    "QS C-Word_Num-Syls<=2{*+x@*,*+1@*,*+2@*}",
    "QS C-Word_Num-Syls<=3{*+x@*,*+1@*,*+2@*,*+3@*}",
    "QS C-Word_Num-Syls<=4{*+x@*,*+1@*,*+2@*,*+3@*,*+4@*}",
    "QS C-Word_Num-Syls<=5{*+x@*,*+1@*,*+2@*,*+3@*,*+4@*,*+5@*}",
    "QS C-Word_Num-Syls<=6{*+x@*,*+1@*,*+2@*,*+3@*,*+4@*,*+5@*,*+6@*}",
    "QS C-Word_Num-Syls<=7{*+x@*,*+1@*,*+2@*,*+3@*,*+4@*,*+5@*,*+6@*,*+7@*}",
]

WORD_POSITION_IN_PHRASE_FW_EQ = [
    "QS Pos_C-Word_in_C-Phrase(Fw)==x{*@x+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==1{*@1+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==2{*@2+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==3{*@3+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==4{*@4+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==5{*@5+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==6{*@6+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==7{*@7+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==8{*@8+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==9{*@9+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==10{*@10+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==11{*@11+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==12{*@12+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)==13{*@13+*}",
]
WORD_POSITION_IN_PHRASE_FW_LTE = [
    "QS Pos_C-Word_in_C-Phrase(Fw)<=1{*@x+*,*@1+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=2{*@x+*,*@1+*,*@2+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=3{*@x+*,*@1+*,*@2+*,*@3+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=4{*@x+*,*@1+*,*@2+*,*@3+*,*@4+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=5{*@x+*,*@1+*,*@2+*,*@3+*,*@4+*,*@5+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=6{*@x+*,*@1+*,*@2+*,*@3+*,*@4+*,*@5+*,*@6+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=7{*@x+*,*@1+*,*@2+*,*@3+*,*@4+*,*@5+*,*@6+*,*@7+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=8{*@x+*,*@1+*,*@2+*,*@3+*,*@4+*,*@5+*,*@6+*,*@7+*,*@8+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=9{*@?+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=10{*@?+*,*@10+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=11{*@?+*,*@10+*,*@11+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=12{*@?+*,*@10+*,*@11+*,*@12+*}",
    "QS Pos_C-Word_in_C-Phrase(Fw)<=13{*@?+*,*@10+*,*@11+*,*@12+*,*@13+*}",
]
WORD_POSITION_IN_PHRASE_BW_EQ = [
    "QS Pos_C-Word_in_C-Phrase(Bw)==x{*+x&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==0{*+0&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==1{*+1&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==2{*+2&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==3{*+3&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==4{*+4&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==5{*+5&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==6{*+6&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==7{*+7&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==8{*+8&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==9{*+9&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==10{*+10&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==11{*+11&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==12{*+12&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)==13{*+13&*}",
]
WORD_POSITION_IN_PHRASE_BW_LTE = [
    "QS Pos_C-Word_in_C-Phrase(Bw)<=1{*+x&*,*+0&*,*+1&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=2{*+x&*,*+0&*,*+1&*,*+2&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=3{*+x&*,*+0&*,*+1&*,*+2&*,*+3&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=4{*+x&*,*+0&*,*+1&*,*+2&*,*+3&*,*+4&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=5{*+x&*,*+0&*,*+1&*,*+2&*,*+3&*,*+4&*,*+5&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=6{*+x&*,*+0&*,*+1&*,*+2&*,*+3&*,*+4&*,*+5&*,*+6&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=7{*+x&*,*+0&*,*+1&*,*+2&*,*+3&*,*+4&*,*+5&*,*+6&*,*+7&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=8{*+x&*,*+0&*,*+1&*,*+2&*,*+3&*,*+4&*,*+5&*,*+6&*,*+7&*,*+8&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=9{*+?&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=10{*+?&*,*+10&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=11{*+?&*,*+10&*,*+11&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=12{*+?&*,*+10&*,*+11&*,*+12&*}",
    "QS Pos_C-Word_in_C-Phrase(Bw)<=13{*+?&*,*+10&*,*+11&*,*+12&*,*+13&*}",
]

NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_EQ = [
    "QS Num-ContWord_before_C-Word_in_C-Phrase==x{*&x+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==1{*&1+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==2{*&2+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==3{*&3+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==4{*&4+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==5{*&5+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==6{*&6+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==7{*&7+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==8{*&8+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase==9{*&9+*}",
]
NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_LTE = [
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=1{*&x+*,*&1+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=2{*&x+*,*&1+*,*&2+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=3{*&x+*,*&1+*,*&2+*,*&3+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=4{*&x+*,*&1+*,*&2+*,*&3+*,*&4+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=5{*&x+*,*&1+*,*&2+*,*&3+*,*&4+*,*&5+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=6{*&x+*,*&1+*,*&2+*,*&3+*,*&4+*,*&5+*,*&6+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=7{*&x+*,*&1+*,*&2+*,*&3+*,*&4+*,*&5+*,*&6+*,*&7+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=8{*&x+*,*&1+*,*&2+*,*&3+*,*&4+*,*&5+*,*&6+*,*&7+*,*&8+*}",
    "QS Num-ContWord_before_C-Word_in_C-Phrase<=9{*&?+*}",
]
NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_EQ = [
    "QS Num-ContWord_after_C-Word_in_C-Phrase==x{*+x#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==0{*+0#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==1{*+1#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==2{*+2#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==3{*+3#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==4{*+4#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==5{*+5#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==6{*+6#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==7{*+7#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase==8{*+8#*}",
]
NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_LTE = [
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=0{*+x#*,*+0#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=1{*+x#*,*+0#*,*+1#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=2{*+x#*,*+0#*,*+1#*,*+2#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=3{*+x#*,*+0#*,*+1#*,*+2#*,*+3#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=4{*+x#*,*+0#*,*+1#*,*+2#*,*+3#*,*+4#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=5{*+x#*,*+0#*,*+1#*,*+2#*,*+3#*,*+4#*,*+5#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=6{*+x#*,*+0#*,*+1#*,*+2#*,*+3#*,*+4#*,*+5#*,*+6#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=7{*+x#*,*+0#*,*+1#*,*+2#*,*+3#*,*+4#*,*+5#*,*+6#*,*+7#*}",
    "QS Num-ContWord_after_C-Word_in_C-Phrase<=8{*+x#*,*+0#*,*+1#*,*+2#*,*+3#*,*+4#*,*+5#*,*+6#*,*+7#*,*+8#*}",
]
NUM_WORDS_FROM_PREV_CONT_WORD_EQ = [
    "QS Num-Words_from_prev-ContWord==x{*#x+*}",
    "QS Num-Words_from_prev-ContWord==0{*#0+*}",
    "QS Num-Words_from_prev-ContWord==1{*#1+*}",
    "QS Num-Words_from_prev-ContWord==2{*#2+*}",
    "QS Num-Words_from_prev-ContWord==3{*#3+*}",
    "QS Num-Words_from_prev-ContWord==4{*#4+*}",
    "QS Num-Words_from_prev-ContWord==5{*#5+*}",
]
NUM_WORDS_FROM_PREV_CONT_WORD_LTE = [
    "QS Num-Words_from_prev-ContWord<=0{*#x+*,*#0+*}",
    "QS Num-Words_from_prev-ContWord<=1{*#x+*,*#0+*,*#1+*}",
    "QS Num-Words_from_prev-ContWord<=2{*#x+*,*#0+*,*#1+*,*#2+*}",
    "QS Num-Words_from_prev-ContWord<=3{*#x+*,*#0+*,*#1+*,*#2+*,*#3+*}",
    "QS Num-Words_from_prev-ContWord<=4{*#x+*,*#0+*,*#1+*,*#2+*,*#3+*,*#4+*}",
    "QS Num-Words_from_prev-ContWord<=5{*#x+*,*#0+*,*#1+*,*#2+*,*#3+*,*#4+*,*#5+*}",
]
NUM_WORDS_FROM_NEXT_CONT_WORD_EQ = [
    "QS Num-Words_from_next-ContWord==x{*+x/F:*}",
    "QS Num-Words_from_next-ContWord==0{*+0/F:*}",
    "QS Num-Words_from_next-ContWord==1{*+1/F:*}",
    "QS Num-Words_from_next-ContWord==2{*+2/F:*}",
    "QS Num-Words_from_next-ContWord==3{*+3/F:*}",
    "QS Num-Words_from_next-ContWord==4{*+4/F:*}",
    "QS Num-Words_from_next-ContWord==5{*+5/F:*}",
]
NUM_WORDS_FROM_NEXT_CONT_WORD_LTE = [
    "QS Num-Words_from_next-ContWord<=0{*+x/F:*,*+0/F:*}",
    "QS Num-Words_from_next-ContWord<=1{*+x/F:*,*+0/F:*,*+1/F:*}",
    "QS Num-Words_from_next-ContWord<=2{*+x/F:*,*+0/F:*,*+1/F:*,*+2/F:*}",
    "QS Num-Words_from_next-ContWord<=3{*+x/F:*,*+0/F:*,*+1/F:*,*+2/F:*,*+3/F:*}",
    "QS Num-Words_from_next-ContWord<=4{*+x/F:*,*+0/F:*,*+1/F:*,*+2/F:*,*+3/F:*,*+4/F:*}",
    "QS Num-Words_from_next-ContWord<=5{*+x/F:*,*+0/F:*,*+1/F:*,*+2/F:*,*+3/F:*,*+4/F:*,*+5/F:*}",
]

WORD_LEN_IN_NO_SYLS_EQ_R = [
    "QS R-Word_Num-Syls==0{*_0/G:*}",
    "QS R-Word_Num-Syls==1{*_1/G:*}",
    "QS R-Word_Num-Syls==2{*_2/G:*}",
    "QS R-Word_Num-Syls==3{*_3/G:*}",
    "QS R-Word_Num-Syls==4{*_4/G:*}",
    "QS R-Word_Num-Syls==5{*_5/G:*}",
    "QS R-Word_Num-Syls==6{*_6/G:*}",
    "QS R-Word_Num-Syls==7{*_7/G:*}",
]
WORD_LEN_IN_NO_SYLS_LTE_R = [
    "QS R-Word_Num-Syls<=1{*_0/G:*,*_1/G:*}",
    "QS R-Word_Num-Syls<=2{*_0/G:*,*_1/G:*,*_2/G:*}",
    "QS R-Word_Num-Syls<=3{*_0/G:*,*_1/G:*,*_2/G:*,*_3/G:*}",
    "QS R-Word_Num-Syls<=4{*_0/G:*,*_1/G:*,*_2/G:*,*_3/G:*,*_4/G:*}",
    "QS R-Word_Num-Syls<=5{*_0/G:*,*_1/G:*,*_2/G:*,*_3/G:*,*_4/G:*,*_5/G:*}",
    "QS R-Word_Num-Syls<=6{*_0/G:*,*_1/G:*,*_2/G:*,*_3/G:*,*_4/G:*,*_5/G:*,*_6/G:*}",
    "QS R-Word_Num-Syls<=7{*_0/G:*,*_1/G:*,*_2/G:*,*_3/G:*,*_4/G:*,*_5/G:*,*_6/G:*,*_7/G:*}",
]

PHRASE_LEN_IN_NO_SYLS_EQ_L = [
    "QS L-Phrase_Num-Syls==0{*/G:0_*}",
    "QS L-Phrase_Num-Syls==1{*/G:1_*}",
    "QS L-Phrase_Num-Syls==2{*/G:2_*}",
    "QS L-Phrase_Num-Syls==3{*/G:3_*}",
    "QS L-Phrase_Num-Syls==4{*/G:4_*}",
    "QS L-Phrase_Num-Syls==5{*/G:5_*}",
    "QS L-Phrase_Num-Syls==6{*/G:6_*}",
    "QS L-Phrase_Num-Syls==7{*/G:7_*}",
    "QS L-Phrase_Num-Syls==8{*/G:8_*}",
    "QS L-Phrase_Num-Syls==9{*/G:9_*}",
    "QS L-Phrase_Num-Syls==10{*/G:10_*}",
    "QS L-Phrase_Num-Syls==11{*/G:11_*}",
    "QS L-Phrase_Num-Syls==12{*/G:12_*}",
    "QS L-Phrase_Num-Syls==13{*/G:13_*}",
    "QS L-Phrase_Num-Syls==14{*/G:14_*}",
    "QS L-Phrase_Num-Syls==15{*/G:15_*}",
    "QS L-Phrase_Num-Syls==16{*/G:16_*}",
    "QS L-Phrase_Num-Syls==17{*/G:17_*}",
    "QS L-Phrase_Num-Syls==18{*/G:18_*}",
    "QS L-Phrase_Num-Syls==19{*/G:19_*}",
    "QS L-Phrase_Num-Syls==20{*/G:20_*}",
]
PHRASE_LEN_IN_NO_SYLS_LTE_L = [
    "QS L-Phrase_Num-Syls<=1{*/G:0_*,*/G:1_*}",
    "QS L-Phrase_Num-Syls<=2{*/G:0_*,*/G:1_*,*/G:2_*}",
    "QS L-Phrase_Num-Syls<=3{*/G:0_*,*/G:1_*,*/G:2_*,*/G:3_*}",
    "QS L-Phrase_Num-Syls<=4{*/G:0_*,*/G:1_*,*/G:2_*,*/G:3_*,*/G:4_*}",
    "QS L-Phrase_Num-Syls<=5{*/G:0_*,*/G:1_*,*/G:2_*,*/G:3_*,*/G:4_*,*/G:5_*}",
    "QS L-Phrase_Num-Syls<=6{*/G:0_*,*/G:1_*,*/G:2_*,*/G:3_*,*/G:4_*,*/G:5_*,*/G:6_*}",
    "QS L-Phrase_Num-Syls<=7{*/G:0_*,*/G:1_*,*/G:2_*,*/G:3_*,*/G:4_*,*/G:5_*,*/G:6_*,*/G:7_*}",
    "QS L-Phrase_Num-Syls<=8{*/G:0_*,*/G:1_*,*/G:2_*,*/G:3_*,*/G:4_*,*/G:5_*,*/G:6_*,*/G:7_*,*/G:8_*}",
    "QS L-Phrase_Num-Syls<=9{*/G:?_*}",
    "QS L-Phrase_Num-Syls<=10{*/G:?_*,*/G:10_*}",
    "QS L-Phrase_Num-Syls<=11{*/G:?_*,*/G:10_*,*/G:11_*}",
    "QS L-Phrase_Num-Syls<=12{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*}",
    "QS L-Phrase_Num-Syls<=13{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*,*/G:13_*}",
    "QS L-Phrase_Num-Syls<=14{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*,*/G:13_*,*/G:14_*}",
    "QS L-Phrase_Num-Syls<=15{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*,*/G:13_*,*/G:14_*,*/G:15_*}",
    "QS L-Phrase_Num-Syls<=16{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*,*/G:13_*,*/G:14_*,*/G:15_*,*/G:16_*}",
    "QS L-Phrase_Num-Syls<=17{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*,*/G:13_*,*/G:14_*,*/G:15_*,*/G:16_*,*/G:17_*}",
    "QS L-Phrase_Num-Syls<=18{*/G:?_*,*/G:10_*,*/G:11_*,*/G:12_*,*/G:13_*,*/G:14_*,*/G:15_*,*/G:16_*,*/G:17_*,*/G:18_*}",
    "QS L-Phrase_Num-Syls<=19{*/G:?_*,*/G:1?_*}",
    "QS L-Phrase_Num-Syls<=20{*/G:?_*,*/G:1?_*,*/G:20_*}",
]

PHRASE_LEN_IN_NO_WORDS_EQ_L = [
    "QS L-Phrase_Num-Words==0{*_0/H:*}",
    "QS L-Phrase_Num-Words==1{*_1/H:*}",
    "QS L-Phrase_Num-Words==2{*_2/H:*}",
    "QS L-Phrase_Num-Words==3{*_3/H:*}",
    "QS L-Phrase_Num-Words==4{*_4/H:*}",
    "QS L-Phrase_Num-Words==5{*_5/H:*}",
    "QS L-Phrase_Num-Words==6{*_6/H:*}",
    "QS L-Phrase_Num-Words==7{*_7/H:*}",
    "QS L-Phrase_Num-Words==8{*_8/H:*}",
    "QS L-Phrase_Num-Words==9{*_9/H:*}",
    "QS L-Phrase_Num-Words==10{*_10/H:*}",
    "QS L-Phrase_Num-Words==11{*_11/H:*}",
    "QS L-Phrase_Num-Words==12{*_12/H:*}",
    "QS L-Phrase_Num-Words==13{*_13/H:*}",
]
PHRASE_LEN_IN_NO_WORDS_LTE_L = [
    "QS L-Phrase_Num-Words<=1{*_0/H:*,*_1/H:*}",
    "QS L-Phrase_Num-Words<=2{*_0/H:*,*_1/H:*,*_2/H:*}",
    "QS L-Phrase_Num-Words<=3{*_0/H:*,*_1/H:*,*_2/H:*,*_3/H:*}",
    "QS L-Phrase_Num-Words<=4{*_0/H:*,*_1/H:*,*_2/H:*,*_3/H:*,*_4/H:*}",
    "QS L-Phrase_Num-Words<=5{*_0/H:*,*_1/H:*,*_2/H:*,*_3/H:*,*_4/H:*,*_5/H:*}",
    "QS L-Phrase_Num-Words<=6{*_0/H:*,*_1/H:*,*_2/H:*,*_3/H:*,*_4/H:*,*_5/H:*,*_6/H:*}",
    "QS L-Phrase_Num-Words<=7{*_0/H:*,*_1/H:*,*_2/H:*,*_3/H:*,*_4/H:*,*_5/H:*,*_6/H:*,*_7/H:*}",
    "QS L-Phrase_Num-Words<=8{*_0/H:*,*_1/H:*,*_2/H:*,*_3/H:*,*_4/H:*,*_5/H:*,*_6/H:*,*_7/H:*,*_8/H:*}",
    "QS L-Phrase_Num-Words<=9{*_?/H:*}",
    "QS L-Phrase_Num-Words<=10{*_?/H:*,*_10/H:*}",
    "QS L-Phrase_Num-Words<=11{*_?/H:*,*_10/H:*,*_11/H:*}",
    "QS L-Phrase_Num-Words<=12{*_?/H:*,*_10/H:*,*_11/H:*,*_12/H:*}",
    "QS L-Phrase_Num-Words<=13{*_?/H:*,*_10/H:*,*_11/H:*,*_12/H:*,*_13/H:*}",
]

PHRASE_LEN_IN_NO_SYLS_EQ_C = [
    "QS C-Phrase_Num-Syls==x{*/H:x=*}",
    "QS C-Phrase_Num-Syls==0{*/H:0=*}",
    "QS C-Phrase_Num-Syls==1{*/H:1=*}",
    "QS C-Phrase_Num-Syls==2{*/H:2=*}",
    "QS C-Phrase_Num-Syls==3{*/H:3=*}",
    "QS C-Phrase_Num-Syls==4{*/H:4=*}",
    "QS C-Phrase_Num-Syls==5{*/H:5=*}",
    "QS C-Phrase_Num-Syls==6{*/H:6=*}",
    "QS C-Phrase_Num-Syls==7{*/H:7=*}",
    "QS C-Phrase_Num-Syls==8{*/H:8=*}",
    "QS C-Phrase_Num-Syls==9{*/H:9=*}",
    "QS C-Phrase_Num-Syls==10{*/H:10=*}",
    "QS C-Phrase_Num-Syls==11{*/H:11=*}",
    "QS C-Phrase_Num-Syls==12{*/H:12=*}",
    "QS C-Phrase_Num-Syls==13{*/H:13=*}",
    "QS C-Phrase_Num-Syls==14{*/H:14=*}",
    "QS C-Phrase_Num-Syls==15{*/H:15=*}",
    "QS C-Phrase_Num-Syls==16{*/H:16=*}",
    "QS C-Phrase_Num-Syls==17{*/H:17=*}",
    "QS C-Phrase_Num-Syls==18{*/H:18=*}",
    "QS C-Phrase_Num-Syls==19{*/H:19=*}",
    "QS C-Phrase_Num-Syls==20{*/H:20=*}",
]
PHRASE_LEN_IN_NO_SYLS_LTE_C = [
    "QS C-Phrase_Num-Syls<=0{*/H:x=*,*/H:0=*}",
    "QS C-Phrase_Num-Syls<=1{*/H:x=*,*/H:0=*,*/H:1=*}",
    "QS C-Phrase_Num-Syls<=2{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*}",
    "QS C-Phrase_Num-Syls<=3{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*,*/H:3=*}",
    "QS C-Phrase_Num-Syls<=4{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*,*/H:3=*,*/H:4=*}",
    "QS C-Phrase_Num-Syls<=5{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*,*/H:3=*,*/H:4=*,*/H:5=*}",
    "QS C-Phrase_Num-Syls<=6{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*,*/H:3=*,*/H:4=*,*/H:5=*,*/H:6=*}",
    "QS C-Phrase_Num-Syls<=7{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*,*/H:3=*,*/H:4=*,*/H:5=*,*/H:6=*,*/H:7=*}",
    "QS C-Phrase_Num-Syls<=8{*/H:x=*,*/H:0=*,*/H:1=*,*/H:2=*,*/H:3=*,*/H:4=*,*/H:5=*,*/H:6=*,*/H:7=*,*/H:8=*}",
    "QS C-Phrase_Num-Syls<=9{*/H:?=*}",
    "QS C-Phrase_Num-Syls<=10{*/H:?=*,*/H:10=*}",
    "QS C-Phrase_Num-Syls<=11{*/H:?=*,*/H:10=*,*/H:11=*}",
    "QS C-Phrase_Num-Syls<=12{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*}",
    "QS C-Phrase_Num-Syls<=13{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*,*/H:13=*}",
    "QS C-Phrase_Num-Syls<=14{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*,*/H:13=*,*/H:14=*}",
    "QS C-Phrase_Num-Syls<=15{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*,*/H:13=*,*/H:14=*,*/H:15=*}",
    "QS C-Phrase_Num-Syls<=16{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*,*/H:13=*,*/H:14=*,*/H:15=*,*/H:16=*}",
    "QS C-Phrase_Num-Syls<=17{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*,*/H:13=*,*/H:14=*,*/H:15=*,*/H:16=*,*/H:17=*}",
    "QS C-Phrase_Num-Syls<=18{*/H:?=*,*/H:10=*,*/H:11=*,*/H:12=*,*/H:13=*,*/H:14=*,*/H:15=*,*/H:16=*,*/H:17=*,*/H:18=*}",
    "QS C-Phrase_Num-Syls<=19{*/H:?=*,*/H:1?=*}",
    "QS C-Phrase_Num-Syls<=20{*/H:?=*,*/H:1?=*,*/H:20=*}",
]

PHRASE_LEN_IN_NO_WORDS_EQ_C = [
    "QS C-Phrase_Num-Words==x{*=x@*}",
    "QS C-Phrase_Num-Words==0{*=0@*}",
    "QS C-Phrase_Num-Words==1{*=1@*}",
    "QS C-Phrase_Num-Words==2{*=2@*}",
    "QS C-Phrase_Num-Words==3{*=3@*}",
    "QS C-Phrase_Num-Words==4{*=4@*}",
    "QS C-Phrase_Num-Words==5{*=5@*}",
    "QS C-Phrase_Num-Words==6{*=6@*}",
    "QS C-Phrase_Num-Words==7{*=7@*}",
    "QS C-Phrase_Num-Words==8{*=8@*}",
    "QS C-Phrase_Num-Words==9{*=9@*}",
    "QS C-Phrase_Num-Words==10{*=10@*}",
    "QS C-Phrase_Num-Words==11{*=11@*}",
    "QS C-Phrase_Num-Words==12{*=12@*}",
    "QS C-Phrase_Num-Words==13{*=13@*}",
]
PHRASE_LEN_IN_NO_WORDS_LTE_C =[
    "QS C-Phrase_Num-Words<=0{*=x@*,*=0@*}",
    "QS C-Phrase_Num-Words<=1{*=x@*,*=0@*,*=1@*}",
    "QS C-Phrase_Num-Words<=2{*=x@*,*=0@*,*=1@*,*=2@*}",
    "QS C-Phrase_Num-Words<=3{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*}",
    "QS C-Phrase_Num-Words<=4{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*}",
    "QS C-Phrase_Num-Words<=5{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*}",
    "QS C-Phrase_Num-Words<=6{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*}",
    "QS C-Phrase_Num-Words<=7{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*}",
    "QS C-Phrase_Num-Words<=8{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*,*=8@*}",
    "QS C-Phrase_Num-Words<=9{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*,*=8@*,*=9@*}",
    "QS C-Phrase_Num-Words<=10{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*,*=8@*,*=9@*,*=10@*}",
    "QS C-Phrase_Num-Words<=11{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*,*=8@*,*=9@*,*=10@*,*=11@*}",
    "QS C-Phrase_Num-Words<=12{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*,*=8@*,*=9@*,*=10@*,*=11@*,*=12@*}",
    "QS C-Phrase_Num-Words<=13{*=x@*,*=0@*,*=1@*,*=2@*,*=3@*,*=4@*,*=5@*,*=6@*,*=7@*,*=8@*,*=9@*,*=10@*,*=11@*,*=12@*,*=13@*}",
]

PHRASE_POSITION_IN_UTTERANCE_FW_EQ_C = [
    "QS Pos_C-Phrase_in_Utterance(Fw)==1{*@1=*}",
    "QS Pos_C-Phrase_in_Utterance(Fw)==2{*@2=*}",
    "QS Pos_C-Phrase_in_Utterance(Fw)==3{*@3=*}",
    "QS Pos_C-Phrase_in_Utterance(Fw)==4{*@4=*}",
]
PHRASE_POSITION_IN_UTTERANCE_FW_LTE_C = [
    "QS Pos_C-Phrase_in_Utterance(Fw)<=2{*@1=*,*@2=*}",
    "QS Pos_C-Phrase_in_Utterance(Fw)<=3{*@1=*,*@2=*,*@3=*}",
    "QS Pos_C-Phrase_in_Utterance(Fw)<=4{*@1=*,*@2=*,*@3=*,*@4=*}",
]

PHRASE_POSITION_IN_UTTERANCE_BW_EQ_C = [
    "QS Pos_C-Phrase_in_Utterance(Bw)==1{*=1|*}",
    "QS Pos_C-Phrase_in_Utterance(Bw)==2{*=2|*}",
    "QS Pos_C-Phrase_in_Utterance(Bw)==3{*=3|*}",
    "QS Pos_C-Phrase_in_Utterance(Bw)==4{*=4|*}",
]
PHRASE_POSITION_IN_UTTERANCE_BW_LTE_C = [
    "QS Pos_C-Phrase_in_Utterance(Bw)<=2{*=1|*,*=2|*}",
    "QS Pos_C-Phrase_in_Utterance(Bw)<=3{*=1|*,*=2|*,*=3|*}",
    "QS Pos_C-Phrase_in_Utterance(Bw)<=4{*=1|*,*=2|*,*=3|*,*=4|*}",
]

PHRASE_LEN_IN_NO_SYLS_EQ_R = [
    "QS R-Phrase_Num-Syls==0{*/I:0=*}",
    "QS R-Phrase_Num-Syls==1{*/I:1=*}",
    "QS R-Phrase_Num-Syls==2{*/I:2=*}",
    "QS R-Phrase_Num-Syls==3{*/I:3=*}",
    "QS R-Phrase_Num-Syls==4{*/I:4=*}",
    "QS R-Phrase_Num-Syls==5{*/I:5=*}",
    "QS R-Phrase_Num-Syls==6{*/I:6=*}",
    "QS R-Phrase_Num-Syls==7{*/I:7=*}",
    "QS R-Phrase_Num-Syls==8{*/I:8=*}",
    "QS R-Phrase_Num-Syls==9{*/I:9=*}",
    "QS R-Phrase_Num-Syls==10{*/I:10=*}",
    "QS R-Phrase_Num-Syls==11{*/I:11=*}",
    "QS R-Phrase_Num-Syls==12{*/I:12=*}",
    "QS R-Phrase_Num-Syls==13{*/I:13=*}",
    "QS R-Phrase_Num-Syls==14{*/I:14=*}",
    "QS R-Phrase_Num-Syls==15{*/I:15=*}",
    "QS R-Phrase_Num-Syls==16{*/I:16=*}",
    "QS R-Phrase_Num-Syls==17{*/I:17=*}",
    "QS R-Phrase_Num-Syls==18{*/I:18=*}",
    "QS R-Phrase_Num-Syls==19{*/I:19=*}",
    "QS R-Phrase_Num-Syls==20{*/I:20=*}",
]
PHRASE_LEN_IN_NO_SYLS_LTE_R = [
    "QS R-Phrase_Num-Syls<=1{*/I:0=*,*/I:1=*}",
    "QS R-Phrase_Num-Syls<=2{*/I:0=*,*/I:1=*,*/I:2=*}",
    "QS R-Phrase_Num-Syls<=3{*/I:0=*,*/I:1=*,*/I:2=*,*/I:3=*}",
    "QS R-Phrase_Num-Syls<=4{*/I:0=*,*/I:1=*,*/I:2=*,*/I:3=*,*/I:4=*}",
    "QS R-Phrase_Num-Syls<=5{*/I:0=*,*/I:1=*,*/I:2=*,*/I:3=*,*/I:4=*,*/I:5=*}",
    "QS R-Phrase_Num-Syls<=6{*/I:0=*,*/I:1=*,*/I:2=*,*/I:3=*,*/I:4=*,*/I:5=*,*/I:6=*}",
    "QS R-Phrase_Num-Syls<=7{*/I:0=*,*/I:1=*,*/I:2=*,*/I:3=*,*/I:4=*,*/I:5=*,*/I:6=*,*/I:7=*}",
    "QS R-Phrase_Num-Syls<=8{*/I:0=*,*/I:1=*,*/I:2=*,*/I:3=*,*/I:4=*,*/I:5=*,*/I:6=*,*/I:7=*,*/I:8=*}",
    "QS R-Phrase_Num-Syls<=9{*/I:?=*}",
    "QS R-Phrase_Num-Syls<=10{*/I:?=*,*/I:10=*}",
    "QS R-Phrase_Num-Syls<=11{*/I:?=*,*/I:10=*,*/I:11=*}",
    "QS R-Phrase_Num-Syls<=12{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*}",
    "QS R-Phrase_Num-Syls<=13{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*,*/I:13=*}",
    "QS R-Phrase_Num-Syls<=14{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*,*/I:13=*,*/I:14=*}",
    "QS R-Phrase_Num-Syls<=15{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*,*/I:13=*,*/I:14=*,*/I:15=*}",
    "QS R-Phrase_Num-Syls<=16{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*,*/I:13=*,*/I:14=*,*/I:15=*,*/I:16=*}",
    "QS R-Phrase_Num-Syls<=17{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*,*/I:13=*,*/I:14=*,*/I:15=*,*/I:16=*,*/I:17=*}",
    "QS R-Phrase_Num-Syls<=18{*/I:?=*,*/I:10=*,*/I:11=*,*/I:12=*,*/I:13=*,*/I:14=*,*/I:15=*,*/I:16=*,*/I:17=*,*/I:18=*}",
    "QS R-Phrase_Num-Syls<=19{*/I:?=*,*/I:1?=*}",
    "QS R-Phrase_Num-Syls<=20{*/I:?=*,*/I:1?=*,*/I:20=*}",
]

PHRASE_LEN_IN_NO_WORDS_EQ_R = [
    "QS R-Phrase_Num-Words==0{*=0/J:*}",
    "QS R-Phrase_Num-Words==1{*=1/J:*}",
    "QS R-Phrase_Num-Words==2{*=2/J:*}",
    "QS R-Phrase_Num-Words==3{*=3/J:*}",
    "QS R-Phrase_Num-Words==4{*=4/J:*}",
    "QS R-Phrase_Num-Words==5{*=5/J:*}",
    "QS R-Phrase_Num-Words==6{*=6/J:*}",
    "QS R-Phrase_Num-Words==7{*=7/J:*}",
    "QS R-Phrase_Num-Words==8{*=8/J:*}",
    "QS R-Phrase_Num-Words==9{*=9/J:*}",
    "QS R-Phrase_Num-Words==10{*=10/J:*}",
    "QS R-Phrase_Num-Words==11{*=11/J:*}",
    "QS R-Phrase_Num-Words==12{*=12/J:*}",
    "QS R-Phrase_Num-Words==13{*=13/J:*}",
    "QS R-Phrase_Num-Words==14{*=14/J:*}",
    "QS R-Phrase_Num-Words==15{*=15/J:*}",
]
PHRASE_LEN_IN_NO_WORDS_LTE_R = [
    "QS R-Phrase_Num-Words<=1{*=0/J:*,*=1/J:*}",
    "QS R-Phrase_Num-Words<=2{*=0/J:*,*=1/J:*,*=2/J:*}",
    "QS R-Phrase_Num-Words<=3{*=0/J:*,*=1/J:*,*=2/J:*,*=3/J:*}",
    "QS R-Phrase_Num-Words<=4{*=0/J:*,*=1/J:*,*=2/J:*,*=3/J:*,*=4/J:*}",
    "QS R-Phrase_Num-Words<=5{*=0/J:*,*=1/J:*,*=2/J:*,*=3/J:*,*=4/J:*,*=5/J:*}",
    "QS R-Phrase_Num-Words<=6{*=0/J:*,*=1/J:*,*=2/J:*,*=3/J:*,*=4/J:*,*=5/J:*,*=6/J:*}",
    "QS R-Phrase_Num-Words<=7{*=0/J:*,*=1/J:*,*=2/J:*,*=3/J:*,*=4/J:*,*=5/J:*,*=6/J:*,*=7/J:*}",
    "QS R-Phrase_Num-Words<=8{*=0/J:*,*=1/J:*,*=2/J:*,*=3/J:*,*=4/J:*,*=5/J:*,*=6/J:*,*=7/J:*,*=8/J:*}",
    "QS R-Phrase_Num-Words<=9{*=?/J:*}",
    "QS R-Phrase_Num-Words<=10{*=?/J:*,*=10/J:*}",
    "QS R-Phrase_Num-Words<=11{*=?/J:*,*=10/J:*,*=11/J:*}",
    "QS R-Phrase_Num-Words<=12{*=?/J:*,*=10/J:*,*=11/J:*,*=12/J:*}",
    "QS R-Phrase_Num-Words<=13{*=?/J:*,*=10/J:*,*=11/J:*,*=12/J:*,*=13/J:*}",
    "QS R-Phrase_Num-Words<=14{*=?/J:*,*=10/J:*,*=11/J:*,*=12/J:*,*=13/J:*,*=14/J:*}",
    "QS R-Phrase_Num-Words<=15{*=?/J:*,*=10/J:*,*=11/J:*,*=12/J:*,*=13/J:*,*=14/J:*,*=15/J:*}",
]

UTTERANCE_LEN_IN_NO_SYLS_EQ = [
    "QS Num-Syls_in_Utterance==1{*/J:1+*}",
    "QS Num-Syls_in_Utterance==2{*/J:2+*}",
    "QS Num-Syls_in_Utterance==3{*/J:3+*}",
    "QS Num-Syls_in_Utterance==4{*/J:4+*}",
    "QS Num-Syls_in_Utterance==5{*/J:5+*}",
    "QS Num-Syls_in_Utterance==6{*/J:6+*}",
    "QS Num-Syls_in_Utterance==7{*/J:7+*}",
    "QS Num-Syls_in_Utterance==8{*/J:8+*}",
    "QS Num-Syls_in_Utterance==9{*/J:9+*}",
    "QS Num-Syls_in_Utterance==10{*/J:10+*}",
    "QS Num-Syls_in_Utterance==11{*/J:11+*}",
    "QS Num-Syls_in_Utterance==12{*/J:12+*}",
    "QS Num-Syls_in_Utterance==13{*/J:13+*}",
    "QS Num-Syls_in_Utterance==14{*/J:14+*}",
    "QS Num-Syls_in_Utterance==15{*/J:15+*}",
    "QS Num-Syls_in_Utterance==16{*/J:16+*}",
    "QS Num-Syls_in_Utterance==17{*/J:17+*}",
    "QS Num-Syls_in_Utterance==18{*/J:18+*}",
    "QS Num-Syls_in_Utterance==19{*/J:19+*}",
    "QS Num-Syls_in_Utterance==20{*/J:20+*}",
    "QS Num-Syls_in_Utterance==21{*/J:21+*}",
    "QS Num-Syls_in_Utterance==22{*/J:22+*}",
    "QS Num-Syls_in_Utterance==23{*/J:23+*}",
    "QS Num-Syls_in_Utterance==24{*/J:24+*}",
    "QS Num-Syls_in_Utterance==25{*/J:25+*}",
    "QS Num-Syls_in_Utterance==26{*/J:26+*}",
    "QS Num-Syls_in_Utterance==27{*/J:27+*}",
    "QS Num-Syls_in_Utterance==28{*/J:28+*}",
]
UTTERANCE_LEN_IN_NO_SYLS_LTE = [
    "QS Num-Syls_in_Utterance<=2{*/J:1+*,*/J:2+*}",
    "QS Num-Syls_in_Utterance<=3{*/J:1+*,*/J:2+*,*/J:3+*}",
    "QS Num-Syls_in_Utterance<=4{*/J:1+*,*/J:2+*,*/J:3+*,*/J:4+*}",
    "QS Num-Syls_in_Utterance<=5{*/J:1+*,*/J:2+*,*/J:3+*,*/J:4+*,*/J:5+*}",
    "QS Num-Syls_in_Utterance<=6{*/J:1+*,*/J:2+*,*/J:3+*,*/J:4+*,*/J:5+*,*/J:6+*}",
    "QS Num-Syls_in_Utterance<=7{*/J:1+*,*/J:2+*,*/J:3+*,*/J:4+*,*/J:5+*,*/J:6+*,*/J:7+*}",
    "QS Num-Syls_in_Utterance<=8{*/J:1+*,*/J:2+*,*/J:3+*,*/J:4+*,*/J:5+*,*/J:6+*,*/J:7+*,*/J:8+*}",
    "QS Num-Syls_in_Utterance<=9{*/J:?+*}",
    "QS Num-Syls_in_Utterance<=10{*/J:?+*,*/J:10+*}",
    "QS Num-Syls_in_Utterance<=11{*/J:?+*,*/J:10+*,*/J:11+*}",
    "QS Num-Syls_in_Utterance<=12{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*}",
    "QS Num-Syls_in_Utterance<=13{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*,*/J:13+*}",
    "QS Num-Syls_in_Utterance<=14{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*,*/J:13+*,*/J:14+*}",
    "QS Num-Syls_in_Utterance<=15{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*,*/J:13+*,*/J:14+*,*/J:15+*}",
    "QS Num-Syls_in_Utterance<=16{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*,*/J:13+*,*/J:14+*,*/J:15+*,*/J:16+*}",
    "QS Num-Syls_in_Utterance<=17{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*,*/J:13+*,*/J:14+*,*/J:15+*,*/J:16+*,*/J:17+*}",
    "QS Num-Syls_in_Utterance<=18{*/J:?+*,*/J:10+*,*/J:11+*,*/J:12+*,*/J:13+*,*/J:14+*,*/J:15+*,*/J:16+*,*/J:17+*,*/J:18+*}",
    "QS Num-Syls_in_Utterance<=19{*/J:?+*,*/J:1?+*}",
    "QS Num-Syls_in_Utterance<=20{*/J:?+*,*/J:1?+*,*/J:20+*}",
    "QS Num-Syls_in_Utterance<=21{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*}",
    "QS Num-Syls_in_Utterance<=22{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*}",
    "QS Num-Syls_in_Utterance<=23{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*,*/J:23+*}",
    "QS Num-Syls_in_Utterance<=24{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*,*/J:23+*,*/J:24+*}",
    "QS Num-Syls_in_Utterance<=25{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*,*/J:23+*,*/J:24+*,*/J:25+*}",
    "QS Num-Syls_in_Utterance<=26{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*,*/J:23+*,*/J:24+*,*/J:25+*,*/J:26+*}",
    "QS Num-Syls_in_Utterance<=27{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*,*/J:23+*,*/J:24+*,*/J:25+*,*/J:26+*,*/J:27+*}",
    "QS Num-Syls_in_Utterance<=28{*/J:?+*,*/J:1?+*,*/J:20+*,*/J:21+*,*/J:22+*,*/J:23+*,*/J:24+*,*/J:25+*,*/J:26+*,*/J:27+*,*/J:28+*}",
]

UTTERANCE_LEN_IN_NO_WORDS_EQ = [
    "QS Num-Words_in_Utterance==1{*+1-*}",
    "QS Num-Words_in_Utterance==2{*+2-*}",
    "QS Num-Words_in_Utterance==3{*+3-*}",
    "QS Num-Words_in_Utterance==4{*+4-*}",
    "QS Num-Words_in_Utterance==5{*+5-*}",
    "QS Num-Words_in_Utterance==6{*+6-*}",
    "QS Num-Words_in_Utterance==7{*+7-*}",
    "QS Num-Words_in_Utterance==8{*+8-*}",
    "QS Num-Words_in_Utterance==9{*+9-*}",
    "QS Num-Words_in_Utterance==10{*+10-*}",
    "QS Num-Words_in_Utterance==11{*+11-*}",
    "QS Num-Words_in_Utterance==12{*+12-*}",
    "QS Num-Words_in_Utterance==13{*+13-*}",
]
UTTERANCE_LEN_IN_NO_WORDS_LTE = [
    "QS Num-Words_in_Utterance<=2{*+1-*,*+2-*}",
    "QS Num-Words_in_Utterance<=3{*+1-*,*+2-*,*+3-*}",
    "QS Num-Words_in_Utterance<=4{*+1-*,*+2-*,*+3-*,*+4-*}",
    "QS Num-Words_in_Utterance<=5{*+1-*,*+2-*,*+3-*,*+4-*,*+5-*}",
    "QS Num-Words_in_Utterance<=6{*+1-*,*+2-*,*+3-*,*+4-*,*+5-*,*+6-*}",
    "QS Num-Words_in_Utterance<=7{*+1-*,*+2-*,*+3-*,*+4-*,*+5-*,*+6-*,*+7-*}",
    "QS Num-Words_in_Utterance<=8{*+1-*,*+2-*,*+3-*,*+4-*,*+5-*,*+6-*,*+7-*,*+8-*}",
    "QS Num-Words_in_Utterance<=9{*+?-*}",
    "QS Num-Words_in_Utterance<=10{*+?-*,*+10-*}",
    "QS Num-Words_in_Utterance<=11{*+?-*,*+10-*,*+11-*}",
    "QS Num-Words_in_Utterance<=12{*+?-*,*+10-*,*+11-*,*+12-*}",
    "QS Num-Words_in_Utterance<=13{*+?-*,*+10-*,*+11-*,*+12-*,*+13-*}",
]

UTTERANCE_LEN_IN_NO_PHRASES_EQ = [
    "QS Num-Phrases_in_Utterance==1{*-1/K:*}",
    "QS Num-Phrases_in_Utterance==2{*-2/K:*}",
    "QS Num-Phrases_in_Utterance==3{*-3/K:*}",
    "QS Num-Phrases_in_Utterance==4{*-4/K:*}",
]
UTTERANCE_LEN_IN_NO_PHRASES_LTE = [
    "QS Num-Phrases_in_Utterance<=2{*-1/K:*,*-2/K:*}",
    "QS Num-Phrases_in_Utterance<=3{*-1/K:*,*-2/K:*,*-3/K:*}",
    "QS Num-Phrases_in_Utterance<=4{*-1/K:*,*-2/K:*,*-3/K:*,*-4/K:*}",
]

###############################################################################
#HIGH-LEVEL FEATURE GROUPS
###############################################################################

###############################################################################
# SEGMENTAL
###############################################################################

QUINTPHONE_SEGMENT_VC = [
    QUINTPHONE_SEGMENT_VC_LL,
    QUINTPHONE_SEGMENT_VC_L,
    QUINTPHONE_SEGMENT_VC_C,
    QUINTPHONE_SEGMENT_VC_R,
    QUINTPHONE_SEGMENT_VC_RR,
]
QUINTPHONE_SEGMENT_ARTICULATORY_TYPE = [
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_LL,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_L,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_C,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_R,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_RR,
]
SEG_POS_IN_SYL = [
    NUM_SEG_IN_SYL_FW_EQ,
    NUM_SEG_IN_SYL_FW_LTE,
    NUM_SEG_IN_SYL_BW_EQ,
    NUM_SEG_IN_SYL_BW_LTE,
]
QUINTPHONE_SEGMENTAL_FEATURES = [
    QUINTPHONE_SEGMENT_VC,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE,
]

QUINTPHONE = [
    QUINTPHONE_SEGMENT_VC,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE,
    SEG_POS_IN_SYL,
    QUINTPHONE_SEGMENTAL_FEATURES,
]

###############################################################################
# SYLLABIC
###############################################################################

SYLLABLE_STRESS = [
    SYL_STRESSED_L,
    SYL_STRESSED_C,
    SYL_STRESSED_R,
]
SYLLABLE_ACCENTED = [
    SYL_ACCENTED_L,
    SYL_ACCENTED_C,
    SYL_ACCENTED_R,
]
SYLLABLE_ACCENT_TYPE = [
    SYL_ACCENT_TYPE_L,
    SYL_ACCENT_TYPE_C,
    SYL_ACCENT_TYPE_R,
]
SYLLABLE_LEN_IN_SEGMENTS = [
    SYL_LEN_IN_NO_SEG_EQ_L,
    SYL_LEN_IN_NO_SEG_LTE_L,
    SYL_LEN_IN_NO_SEG_EQ_C,
    SYL_LEN_IN_NO_SEG_LTE_C,
    SYL_LEN_IN_NO_SEG_EQ_R,
    SYL_LEN_IN_NO_SEG_LTE_R,

]
SYLLABLE_NUCLEUS = [
    SYL_VOWEL_TYPE,
]

SYLLABLE_POSITION_IN_WORD_FW = [
    SYL_POSITION_IN_WORD_FW_EQ,
    SYL_POSITION_IN_WORD_FW_LTE,
]
SYLLABLE_POSITION_IN_WORD_BW = [
    SYL_POSITION_IN_WORD_BW_EQ,
    SYL_POSITION_IN_WORD_BW_LTE,
]
SYLLABLE_POSITION_IN_PHRASE_FW = [
    SYL_POSITION_IN_PHRASE_FW_EQ,
    SYL_POSITION_IN_PHRASE_FW_LTE,
]
SYLLABLE_POSITION_IN_PHRASE_BW = [
    SYL_POSITION_IN_PHRASE_BW_EQ,
    SYL_POSITION_IN_PHRASE_BW_LTE,
]
SYLLABLE_POSITION_IN_WORD = [
    SYLLABLE_POSITION_IN_WORD_FW,
    SYLLABLE_POSITION_IN_WORD_BW,
]
SYLLABLE_POSITION_IN_PHRASE = [
    SYLLABLE_POSITION_IN_PHRASE_FW,
    SYLLABLE_POSITION_IN_PHRASE_BW,
]

SYLLABLE_POSITION = [
    SYLLABLE_POSITION_IN_WORD,
    SYLLABLE_POSITION_IN_PHRASE,
]

NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL = [
    NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
]
NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL = [
    NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
]
NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION = [
    NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL,
    NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL,
]
NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL = [
    NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
]
NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL = [
    NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
]
NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION = [
    NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL,
    NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL,
]

SYLLABLE_NEIGHBOURHOOD = [
    NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL,
    NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL,
    NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION,
    NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL,
    NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL,
    NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION,
]

POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_STRESSED_SYL = [
    NUM_SYLS_FROM_NEXT_STRESSED_SYL_EQ,
    NUM_SYLS_FROM_NEXT_STRESSED_SYL_LTE,
]
POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_STRESSED_SYL = [
    NUM_SYLS_FROM_PREV_STRESSED_SYL_EQ,
    NUM_SYLS_FROM_PREV_STRESSED_SYL_LTE
]
POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_STRESSED_SYLS = [
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_STRESSED_SYL,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_STRESSED_SYL,
]
POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_ACCENTED_SYL = [
    NUM_SYLS_FROM_PREV_ACCENTED_SYL_EQ,
    NUM_SYLS_FROM_PREV_ACCENTED_SYL_LTE,
]
POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_ACCENTED_SYL = [
    NUM_SYLS_FROM_NEXT_ACCENTED_SYL_EQ,
    NUM_SYLS_FROM_NEXT_ACCENTED_SYL_LTE,
]
POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_ACCENTED_SYLS = [
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_ACCENTED_SYL,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_ACCENTED_SYL
]

RELATIVE_SYLLABLE_POSITION = [
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_STRESSED_SYL,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_STRESSED_SYL,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_STRESSED_SYLS,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_ACCENTED_SYL,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_ACCENTED_SYL,
    POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_ACCENTED_SYLS,
]

###############################################################################
# WORD
###############################################################################

WORD_LEN_IN_SYLS = [
    WORD_LEN_IN_NO_SYLS_EQ_R,
    WORD_LEN_IN_NO_SYLS_EQ_C,
    WORD_LEN_IN_NO_SYLS_EQ_L,
    WORD_LEN_IN_NO_SYLS_LTE_R,
    WORD_LEN_IN_NO_SYLS_LTE_C,
    WORD_LEN_IN_NO_SYLS_LTE_L,
]
WORD_POSITION_IN_PHRASE_FW = [
    WORD_POSITION_IN_PHRASE_FW_EQ,
    WORD_POSITION_IN_PHRASE_FW_LTE,
]
WORD_POSITION_IN_PHRASE_BW = [
    WORD_POSITION_IN_PHRASE_BW_EQ,
    WORD_POSITION_IN_PHRASE_BW_LTE,
]
WORD_POSITION_IN_PHRASE = [
    WORD_POSITION_IN_PHRASE_FW,
    WORD_POSITION_IN_PHRASE_BW,
]
WORD_POSITION = [
    WORD_POSITION_IN_PHRASE_FW,
    WORD_POSITION_IN_PHRASE_BW,
    WORD_POSITION_IN_PHRASE,
]
NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE = [
    NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_EQ,
    NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_LTE,
]
NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE = [
    NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_EQ,
    NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_LTE,
]
WORD_SURROUNDINGS = [
    NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE,
    NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE,
]
NUM_WORDS_FROM_PREV_CONT_WORD = [
    NUM_WORDS_FROM_PREV_CONT_WORD_EQ,
    NUM_WORDS_FROM_PREV_CONT_WORD_LTE,
]
NUM_WORDS_FROM_NEXT_CONT_WORD = [
    NUM_WORDS_FROM_NEXT_CONT_WORD_EQ,
    NUM_WORDS_FROM_NEXT_CONT_WORD_LTE,
]
RELATIVE_WORD_POSITION = [
    NUM_WORDS_FROM_PREV_CONT_WORD,
    NUM_WORDS_FROM_NEXT_CONT_WORD,
]

###############################################################################
# PHRASE
###############################################################################

PHRASE_LEN_IN_SYLS = [
    PHRASE_LEN_IN_NO_SYLS_EQ_L,
    PHRASE_LEN_IN_NO_SYLS_LTE_L,
    PHRASE_LEN_IN_NO_SYLS_EQ_C,
    PHRASE_LEN_IN_NO_SYLS_LTE_C,
    PHRASE_LEN_IN_NO_SYLS_EQ_R,
    PHRASE_LEN_IN_NO_SYLS_LTE_R,
]
PHRASE_LEN_IN_WORDS = [
    PHRASE_LEN_IN_NO_WORDS_EQ_L,
    PHRASE_LEN_IN_NO_WORDS_EQ_C,
    PHRASE_LEN_IN_NO_WORDS_EQ_R,
    PHRASE_LEN_IN_NO_WORDS_LTE_L,
    PHRASE_LEN_IN_NO_WORDS_LTE_C,
    PHRASE_LEN_IN_NO_WORDS_LTE_R,
]
PHRASE_LEN = [
    PHRASE_LEN_IN_SYLS,
    PHRASE_LEN_IN_WORDS,
]
PHRASE_POSITION_IN_UTTERANCE_FW = [
    PHRASE_POSITION_IN_UTTERANCE_FW_EQ_C,
    PHRASE_POSITION_IN_UTTERANCE_FW_LTE_C,
]
PHRASE_POSITION_IN_UTTERANCE_BW = [
    PHRASE_POSITION_IN_UTTERANCE_BW_EQ_C,
    PHRASE_POSITION_IN_UTTERANCE_BW_LTE_C,
]
PHRASE_POSITION = [
    PHRASE_POSITION_IN_UTTERANCE_FW,
    PHRASE_POSITION_IN_UTTERANCE_BW,
]

###############################################################################
# UTTERANCE
###############################################################################

UTTERANCE_LEN_IN_SYLS = [
    UTTERANCE_LEN_IN_NO_SYLS_EQ,
    UTTERANCE_LEN_IN_NO_SYLS_LTE,
]
UTTERANCE_LEN_IN_WORDS = [
    UTTERANCE_LEN_IN_NO_WORDS_EQ,
    UTTERANCE_LEN_IN_NO_WORDS_LTE,
]
UTTERANCE_LEN_IN_PHRASES = [
    UTTERANCE_LEN_IN_NO_PHRASES_EQ,
    UTTERANCE_LEN_IN_NO_PHRASES_LTE,
]
UTTERANCE_LEN = [
    UTTERANCE_LEN_IN_SYLS,
    UTTERANCE_LEN_IN_WORDS,
    UTTERANCE_LEN_IN_PHRASES,
]

# SEGMENTAL

SEGMENTAL_POSITIONAL_ABSOLUTE = [
    NUM_SEG_IN_SYL_FW_EQ,
    NUM_SEG_IN_SYL_FW_LTE,
    NUM_SEG_IN_SYL_BW_EQ,
    NUM_SEG_IN_SYL_BW_LTE,
]

SEGMENTAL_QUALITATIVE = [
    QUINTPHONE_SEGMENT_VC_LL,
    QUINTPHONE_SEGMENT_VC_L,
    QUINTPHONE_SEGMENT_VC_C,
    QUINTPHONE_SEGMENT_VC_R,
    QUINTPHONE_SEGMENT_VC_RR,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_LL,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_L,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_C,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_R,
    QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_RR,
]

# SYLLABIC

SYLLABIC_POSITIONAL_ABSOLUTE = [
    SYL_POSITION_IN_WORD_FW_EQ,
    SYL_POSITION_IN_WORD_FW_LTE,
    SYL_POSITION_IN_WORD_BW_EQ,
    SYL_POSITION_IN_WORD_BW_LTE,
    SYL_POSITION_IN_PHRASE_FW_EQ,
    SYL_POSITION_IN_PHRASE_FW_LTE,
    SYL_POSITION_IN_PHRASE_BW_EQ,
    SYL_POSITION_IN_PHRASE_BW_LTE,
]

SYLLABIC_POSITIONAL_RELATIVE = [
    NUM_SYLS_FROM_PREV_STRESSED_SYL_EQ,
    NUM_SYLS_FROM_PREV_STRESSED_SYL_LTE,
    NUM_SYLS_FROM_NEXT_STRESSED_SYL_EQ,
    NUM_SYLS_FROM_NEXT_STRESSED_SYL_LTE,
    NUM_SYLS_FROM_PREV_ACCENTED_SYL_EQ,
    NUM_SYLS_FROM_PREV_ACCENTED_SYL_LTE,
    NUM_SYLS_FROM_NEXT_ACCENTED_SYL_EQ,
    NUM_SYLS_FROM_NEXT_ACCENTED_SYL_LTE,
]

SYLLABIC_QUALITATIVE = [
    SYL_STRESSED_L,
    SYL_ACCENTED_L,
    SYL_ACCENT_TYPE_L,
    SYL_STRESSED_C,
    SYL_ACCENTED_C,
    SYL_ACCENT_TYPE_C,
    SYL_VOWEL_TYPE,
    SYL_STRESSED_R,
    SYL_ACCENTED_R,
    SYL_ACCENT_TYPE_R,
]

SYLLABIC_COMPOSITIONAL = [
    SYL_LEN_IN_NO_SEG_EQ_L,
    SYL_LEN_IN_NO_SEG_LTE_L,
    SYL_LEN_IN_NO_SEG_EQ_C,
    SYL_LEN_IN_NO_SEG_LTE_C,
    SYL_LEN_IN_NO_SEG_EQ_R,
    SYL_LEN_IN_NO_SEG_LTE_R,
]

SYLLABIC_PARENTAL_COMPOSITION = [
    NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
]

# WORD

WORD_POSITIONAL_ABSOLUTE = [
    WORD_POSITION_IN_PHRASE_FW_EQ,
    WORD_POSITION_IN_PHRASE_FW_LTE,
    WORD_POSITION_IN_PHRASE_BW_EQ,
    WORD_POSITION_IN_PHRASE_BW_LTE,
]

WORD_POSITIONAL_RELATIVE = [
    NUM_WORDS_FROM_PREV_CONT_WORD_EQ,
    NUM_WORDS_FROM_PREV_CONT_WORD_LTE,
    NUM_WORDS_FROM_NEXT_CONT_WORD_EQ,
    NUM_WORDS_FROM_NEXT_CONT_WORD_LTE,
]

WORD_COMPOSITIONAL = [
    WORD_LEN_IN_NO_SYLS_EQ_L,
    WORD_LEN_IN_NO_SYLS_LTE_L,
    WORD_LEN_IN_NO_SYLS_EQ_C,
    WORD_LEN_IN_NO_SYLS_LTE_C,
    WORD_LEN_IN_NO_SYLS_EQ_R,
    WORD_LEN_IN_NO_SYLS_LTE_R,
]

WORD_PARENTAL_COMPOSITION = [
    NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_EQ,
    NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_LTE,
    NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_EQ,
    NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_LTE,
]

# PHRASAL

PHRASAL_POSITIONAL_ABSOLUTE = [
    PHRASE_POSITION_IN_UTTERANCE_FW_EQ_C,
    PHRASE_POSITION_IN_UTTERANCE_FW_LTE_C,
    PHRASE_POSITION_IN_UTTERANCE_BW_EQ_C,
    PHRASE_POSITION_IN_UTTERANCE_BW_LTE_C,
]

PHRASAL_COMPOSITIONAL = [
    PHRASE_LEN_IN_NO_SYLS_EQ_L,
    PHRASE_LEN_IN_NO_SYLS_LTE_L,
    PHRASE_LEN_IN_NO_WORDS_EQ_L,
    PHRASE_LEN_IN_NO_WORDS_LTE_L,
    PHRASE_LEN_IN_NO_SYLS_EQ_C,
    PHRASE_LEN_IN_NO_SYLS_LTE_C,
    PHRASE_LEN_IN_NO_WORDS_EQ_C,
    PHRASE_LEN_IN_NO_WORDS_LTE_C,
    PHRASE_LEN_IN_NO_SYLS_EQ_R,
    PHRASE_LEN_IN_NO_SYLS_LTE_R,
    PHRASE_LEN_IN_NO_WORDS_EQ_R,
    PHRASE_LEN_IN_NO_WORDS_LTE_R,
]

# UTTERANCE

UTTERANCE_COMPOSITIONAL = [
    UTTERANCE_LEN_IN_NO_SYLS_EQ,
    UTTERANCE_LEN_IN_NO_SYLS_LTE,
    UTTERANCE_LEN_IN_NO_WORDS_EQ,
    UTTERANCE_LEN_IN_NO_WORDS_LTE,
    UTTERANCE_LEN_IN_NO_PHRASES_EQ,
    UTTERANCE_LEN_IN_NO_PHRASES_LTE,
]

POSITIONAL_ABSOLUTE = [
    SEGMENTAL_POSITIONAL_ABSOLUTE,
    SYLLABIC_POSITIONAL_ABSOLUTE,
    WORD_POSITIONAL_ABSOLUTE,
    PHRASAL_POSITIONAL_ABSOLUTE,
]

POSITIONAL_RELATIVE = [
    SYLLABIC_POSITIONAL_RELATIVE,
    WORD_POSITIONAL_RELATIVE,
]

QUALITATIVE = [
    SEGMENTAL_QUALITATIVE,
    SYLLABIC_QUALITATIVE,
]

COMPOSITIONAL = [
    SYLLABIC_COMPOSITIONAL,
    WORD_COMPOSITIONAL,
    PHRASAL_COMPOSITIONAL,
    UTTERANCE_COMPOSITIONAL,
]

PARENTAL_COMPOSITION = [
    SYLLABIC_PARENTAL_COMPOSITION,
    WORD_PARENTAL_COMPOSITION,
]

SEGMENTAL = [
    SEGMENTAL_POSITIONAL_ABSOLUTE,
    SEGMENTAL_QUALITATIVE,
]

SYLLABIC = [
    SYLLABIC_POSITIONAL_ABSOLUTE,
    SYLLABIC_POSITIONAL_RELATIVE,
    SYLLABIC_QUALITATIVE,
    SYLLABIC_COMPOSITIONAL,
    SYLLABIC_PARENTAL_COMPOSITION,
]

WORD = [
    WORD_POSITIONAL_ABSOLUTE,
    WORD_POSITIONAL_RELATIVE,
    WORD_COMPOSITIONAL,
    WORD_PARENTAL_COMPOSITION,
]

PHRASAL = [
    PHRASAL_POSITIONAL_ABSOLUTE,
    PHRASAL_COMPOSITIONAL,
]

UTTERANCE = [
    UTTERANCE_COMPOSITIONAL,
]

###############################################################################
# TARGET FEATURE GROUPS
###############################################################################
FEATURE_GROUPS = {
    'VUV': VUV,
    'QUINTPHONE_SEGMENT_VC': QUINTPHONE_SEGMENT_VC,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE,
    'SEG_POS_IN_SYL': SEG_POS_IN_SYL,
    'QUINTPHONE_SEGMENTAL_FEATURES': QUINTPHONE_SEGMENTAL_FEATURES,
    'SYLLABLE_STRESS': SYLLABLE_STRESS,
    'SYLLABLE_ACCENTED': SYLLABLE_ACCENTED,
    'SYLLABLE_ACCENT_TYPE': SYLLABLE_ACCENT_TYPE,
    'SYLLABLE_LEN_IN_SEGMENTS': SYLLABLE_LEN_IN_SEGMENTS,
    'SYLLABLE_NUCLEUS': SYLLABLE_NUCLEUS,
    'SYLLABLE_POSITION': SYLLABLE_POSITION,
    'RELATIVE_SYLLABLE_POSITION': RELATIVE_SYLLABLE_POSITION,
    'SYLLABLE_NEIGHBOURHOOD': SYLLABLE_NEIGHBOURHOOD,
    'WORD_LEN_IN_SYLS': WORD_LEN_IN_SYLS,
    'WORD_POSITION': WORD_POSITION,
    'WORD_SURROUNDINGS': WORD_SURROUNDINGS,
    'RELATIVE_WORD_POSITION': RELATIVE_WORD_POSITION,
    'PHRASE_LEN': PHRASE_LEN,
    'PHRASE_POSITION': PHRASE_POSITION,
    'UTTERANCE_LEN': UTTERANCE_LEN,
}

DETAILED_GROUPS = {
    'VUV': VUV,
    'SYL_VOWEL_TYPE': SYL_VOWEL_TYPE,
    'QUINTPHONE_SEGMENT_VC': QUINTPHONE_SEGMENT_VC,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE,
    'SEG_POS_IN_SYL': SEG_POS_IN_SYL,
    'QUINTPHONE_SEGMENTAL_FEATURES': QUINTPHONE_SEGMENTAL_FEATURES,
    'SYLLABLE_STRESS': SYLLABLE_STRESS,
    'SYLLABLE_ACCENTED': SYLLABLE_ACCENTED,
    'SYLLABLE_ACCENT_TYPE': SYLLABLE_ACCENT_TYPE,
    'SYLLABLE_LEN_IN_SEGMENTS': SYLLABLE_LEN_IN_SEGMENTS,
    'SYLLABLE_NUCLEUS': SYLLABLE_NUCLEUS,
    'SYLLABLE_POSITION_IN_WORD_FW': SYLLABLE_POSITION_IN_WORD_FW,
    'SYLLABLE_POSITION_IN_WORD_BW': SYLLABLE_POSITION_IN_WORD_BW,
    'SYLLABLE_POSITION_IN_PHRASE_FW': SYLLABLE_POSITION_IN_PHRASE_FW,
    'SYLLABLE_POSITION_IN_PHRASE_BW': SYLLABLE_POSITION_IN_PHRASE_BW,
    'SYLLABLE_POSITION_IN_WORD': SYLLABLE_POSITION_IN_WORD,
    'SYLLABLE_POSITION_IN_PHRASE': SYLLABLE_POSITION_IN_PHRASE,
    'NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL': NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL,
    'NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL': NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL,
    'NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION': NUM_STRESSED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION,
    'NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL': NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_BEFORE_CURRENT_SYL,
    'NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL': NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_AFTER_CURRENT_SYL,
    'NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION': NUM_ACCENTED_SYLS_IN_CURRENT_PHRASE_IN_RELATION_TO_CURRENT_SYL_POSITION,
    'POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_STRESSED_SYL': POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_STRESSED_SYL,
    'POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_STRESSED_SYL': POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_STRESSED_SYL,
    'POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_STRESSED_SYLS': POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_STRESSED_SYLS,
    'POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_ACCENTED_SYL': POSITION_OF_CURRENT_SYL_IN_RELATION_TO_PREV_ACCENTED_SYL,
    'POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_ACCENTED_SYL': POSITION_OF_CURRENT_SYL_IN_RELATION_TO_NEXT_ACCENTED_SYL,
    'POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_ACCENTED_SYLS': POSITION_OF_CURRENT_SYL_IN_RELATION_TO_OTHER_NEIGHBOURING_ACCENTED_SYLS,
    'WORD_LEN_IN_SYLS': WORD_LEN_IN_SYLS,
    'WORD_POSITION_IN_PHRASE_FW': WORD_POSITION_IN_PHRASE_FW,
    'WORD_POSITION_IN_PHRASE_BW': WORD_POSITION_IN_PHRASE_BW,
    'WORD_POSITION_IN_PHRASE': WORD_POSITION_IN_PHRASE,
    'NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE': NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE,
    'NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE': NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE,
    'NUM_WORDS_FROM_PREV_CONT_WORD': NUM_WORDS_FROM_PREV_CONT_WORD,
    'NUM_WORDS_FROM_NEXT_CONT_WORD': NUM_WORDS_FROM_NEXT_CONT_WORD,
    'PHRASE_LEN_IN_SYLS': PHRASE_LEN_IN_SYLS,
    'PHRASE_LEN_IN_WORDS': PHRASE_LEN_IN_WORDS,
    'PHRASE_POSITION_IN_UTTERANCE_FW': PHRASE_POSITION_IN_UTTERANCE_FW,
    'PHRASE_POSITION_IN_UTTERANCE_BW': PHRASE_POSITION_IN_UTTERANCE_BW,
    'UTTERANCE_LEN_IN_SYLS': UTTERANCE_LEN_IN_SYLS,
    'UTTERANCE_LEN_IN_WORDS': UTTERANCE_LEN_IN_WORDS,
    'UTTERANCE_LEN_IN_PHRASES': UTTERANCE_LEN_IN_PHRASES,
}

ALL_GROUPS = {
    'VUV': VUV,
    'QUINTPHONE_SEGMENT_VC_LL': QUINTPHONE_SEGMENT_VC_LL,
    'QUINTPHONE_SEGMENT_VC_L': QUINTPHONE_SEGMENT_VC_L,
    'QUINTPHONE_SEGMENT_VC_C': QUINTPHONE_SEGMENT_VC_C,
    'QUINTPHONE_SEGMENT_VC_R': QUINTPHONE_SEGMENT_VC_R,
    'QUINTPHONE_SEGMENT_VC_RR': QUINTPHONE_SEGMENT_VC_RR,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_LL': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_LL,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_L': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_L,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_C': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_C,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_R': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_R,
    'QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_RR': QUINTPHONE_SEGMENT_ARTICULATORY_TYPE_RR,
    'NUM_SEG_IN_SYL_FW_EQ': NUM_SEG_IN_SYL_FW_EQ,
    'NUM_SEG_IN_SYL_FW_LTE': NUM_SEG_IN_SYL_FW_LTE,
    'NUM_SEG_IN_SYL_BW_EQ': NUM_SEG_IN_SYL_BW_EQ,
    'NUM_SEG_IN_SYL_BW_LTE': NUM_SEG_IN_SYL_BW_LTE,
    'SYL_STRESSED_L': SYL_STRESSED_L,
    'SYL_ACCENTED_L': SYL_ACCENTED_L,
    'SYL_ACCENT_TYPE_L': SYL_ACCENT_TYPE_L,
    'SYL_LEN_IN_NO_SEG_EQ_L': SYL_LEN_IN_NO_SEG_EQ_L,
    'SYL_LEN_IN_NO_SEG_LTE_L': SYL_LEN_IN_NO_SEG_LTE_L,
    'SYL_STRESSED_C': SYL_STRESSED_C,
    'SYL_ACCENTED_C': SYL_ACCENTED_C,
    'SYL_ACCENT_TYPE_C': SYL_ACCENT_TYPE_C,
    'SYL_LEN_IN_NO_SEG_EQ_C': SYL_LEN_IN_NO_SEG_EQ_C,
    'SYL_LEN_IN_NO_SEG_LTE_C': SYL_LEN_IN_NO_SEG_LTE_C,
    'SYL_POSITION_IN_WORD_FW_EQ': SYL_POSITION_IN_WORD_FW_EQ,
    'SYL_POSITION_IN_WORD_FW_LTE': SYL_POSITION_IN_WORD_FW_LTE,
    'SYL_POSITION_IN_WORD_BW_EQ': SYL_POSITION_IN_WORD_BW_EQ,
    'SYL_POSITION_IN_WORD_BW_LTE': SYL_POSITION_IN_WORD_BW_LTE,
    'SYL_POSITION_IN_PHRASE_FW_EQ': SYL_POSITION_IN_PHRASE_FW_EQ,
    'SYL_POSITION_IN_PHRASE_FW_LTE': SYL_POSITION_IN_PHRASE_FW_LTE,
    'SYL_POSITION_IN_PHRASE_BW_EQ': SYL_POSITION_IN_PHRASE_BW_EQ,
    'SYL_POSITION_IN_PHRASE_BW_LTE': SYL_POSITION_IN_PHRASE_BW_LTE,
    'NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ': NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    'NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE': NUM_STRESSED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    'NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ': NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    'NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE': NUM_STRESSED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    'NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ': NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    'NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE': NUM_ACCENTED_SYLS_BEFORE_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    'NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ': NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_EQ,
    'NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE': NUM_ACCENTED_SYLS_AFTER_CURRENT_SYL_IN_CURRENT_PHRASE_LTE,
    'NUM_SYLS_FROM_PREV_STRESSED_SYL_EQ': NUM_SYLS_FROM_PREV_STRESSED_SYL_EQ,
    'NUM_SYLS_FROM_PREV_STRESSED_SYL_LTE': NUM_SYLS_FROM_PREV_STRESSED_SYL_LTE,
    'NUM_SYLS_FROM_NEXT_STRESSED_SYL_EQ': NUM_SYLS_FROM_NEXT_STRESSED_SYL_EQ,
    'NUM_SYLS_FROM_NEXT_STRESSED_SYL_LTE': NUM_SYLS_FROM_NEXT_STRESSED_SYL_LTE,
    'NUM_SYLS_FROM_PREV_ACCENTED_SYL_EQ': NUM_SYLS_FROM_PREV_ACCENTED_SYL_EQ,
    'NUM_SYLS_FROM_PREV_ACCENTED_SYL_LTE': NUM_SYLS_FROM_PREV_ACCENTED_SYL_LTE,
    'NUM_SYLS_FROM_NEXT_ACCENTED_SYL_EQ': NUM_SYLS_FROM_NEXT_ACCENTED_SYL_EQ,
    'NUM_SYLS_FROM_NEXT_ACCENTED_SYL_LTE': NUM_SYLS_FROM_NEXT_ACCENTED_SYL_LTE,
    'SYL_VOWEL_TYPE': SYL_VOWEL_TYPE,
    'SYL_STRESSED_R': SYL_STRESSED_R,
    'SYL_ACCENTED_R': SYL_ACCENTED_R,
    'SYL_ACCENT_TYPE_R': SYL_ACCENT_TYPE_R,
    'SYL_LEN_IN_NO_SEG_EQ_R': SYL_LEN_IN_NO_SEG_EQ_R,
    'SYL_LEN_IN_NO_SEG_LTE_R': SYL_LEN_IN_NO_SEG_LTE_R,
    'WORD_LEN_IN_NO_SYLS_EQ_L': WORD_LEN_IN_NO_SYLS_EQ_L,
    'WORD_LEN_IN_NO_SYLS_LTE_L': WORD_LEN_IN_NO_SYLS_LTE_L,
    'WORD_LEN_IN_NO_SYLS_EQ_C': WORD_LEN_IN_NO_SYLS_EQ_C,
    'WORD_LEN_IN_NO_SYLS_LTE_C': WORD_LEN_IN_NO_SYLS_LTE_C,
    'WORD_POSITION_IN_PHRASE_FW_EQ': WORD_POSITION_IN_PHRASE_FW_EQ,
    'WORD_POSITION_IN_PHRASE_FW_LTE': WORD_POSITION_IN_PHRASE_FW_LTE,
    'WORD_POSITION_IN_PHRASE_BW_EQ': WORD_POSITION_IN_PHRASE_BW_EQ,
    'WORD_POSITION_IN_PHRASE_BW_LTE': WORD_POSITION_IN_PHRASE_BW_LTE,
    'NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_EQ': NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_EQ,
    'NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_LTE': NUM_CONT_WORDS_BEFORE_CURRENT_WORD_IN_PHRASE_LTE,
    'NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_EQ': NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_EQ,
    'NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_LTE': NUM_CONT_WORDS_AFTER_CURRENT_WORD_IN_PHRASE_LTE,
    'NUM_WORDS_FROM_PREV_CONT_WORD_EQ': NUM_WORDS_FROM_PREV_CONT_WORD_EQ,
    'NUM_WORDS_FROM_PREV_CONT_WORD_LTE': NUM_WORDS_FROM_PREV_CONT_WORD_LTE,
    'NUM_WORDS_FROM_NEXT_CONT_WORD_EQ': NUM_WORDS_FROM_NEXT_CONT_WORD_EQ,
    'NUM_WORDS_FROM_NEXT_CONT_WORD_LTE': NUM_WORDS_FROM_NEXT_CONT_WORD_LTE,
    'WORD_LEN_IN_NO_SYLS_EQ_R': WORD_LEN_IN_NO_SYLS_EQ_R,
    'WORD_LEN_IN_NO_SYLS_LTE_R': WORD_LEN_IN_NO_SYLS_LTE_R,
    'PHRASE_LEN_IN_NO_SYLS_EQ_L': PHRASE_LEN_IN_NO_SYLS_EQ_L,
    'PHRASE_LEN_IN_NO_SYLS_LTE_L': PHRASE_LEN_IN_NO_SYLS_LTE_L,
    'PHRASE_LEN_IN_NO_WORDS_EQ_L': PHRASE_LEN_IN_NO_WORDS_EQ_L,
    'PHRASE_LEN_IN_NO_WORDS_LTE_L': PHRASE_LEN_IN_NO_WORDS_LTE_L,
    'PHRASE_LEN_IN_NO_SYLS_EQ_C': PHRASE_LEN_IN_NO_SYLS_EQ_C,
    'PHRASE_LEN_IN_NO_SYLS_LTE_C': PHRASE_LEN_IN_NO_SYLS_LTE_C,
    'PHRASE_LEN_IN_NO_WORDS_EQ_C': PHRASE_LEN_IN_NO_WORDS_EQ_C,
    'PHRASE_LEN_IN_NO_WORDS_LTE_C': PHRASE_LEN_IN_NO_WORDS_LTE_C,
    'PHRASE_POSITION_IN_UTTERANCE_FW_EQ_C': PHRASE_POSITION_IN_UTTERANCE_FW_EQ_C,
    'PHRASE_POSITION_IN_UTTERANCE_FW_LTE_C': PHRASE_POSITION_IN_UTTERANCE_FW_LTE_C,
    'PHRASE_POSITION_IN_UTTERANCE_BW_EQ_C': PHRASE_POSITION_IN_UTTERANCE_BW_EQ_C,
    'PHRASE_POSITION_IN_UTTERANCE_BW_LTE_C': PHRASE_POSITION_IN_UTTERANCE_BW_LTE_C,
    'PHRASE_LEN_IN_NO_SYLS_EQ_R': PHRASE_LEN_IN_NO_SYLS_EQ_R,
    'PHRASE_LEN_IN_NO_SYLS_LTE_R': PHRASE_LEN_IN_NO_SYLS_LTE_R,
    'PHRASE_LEN_IN_NO_WORDS_EQ_R': PHRASE_LEN_IN_NO_WORDS_EQ_R,
    'PHRASE_LEN_IN_NO_WORDS_LTE_R': PHRASE_LEN_IN_NO_WORDS_LTE_R,
    'UTTERANCE_LEN_IN_NO_SYLS_EQ': UTTERANCE_LEN_IN_NO_SYLS_EQ,
    'UTTERANCE_LEN_IN_NO_SYLS_LTE': UTTERANCE_LEN_IN_NO_SYLS_LTE,
    'UTTERANCE_LEN_IN_NO_WORDS_EQ': UTTERANCE_LEN_IN_NO_WORDS_EQ,
    'UTTERANCE_LEN_IN_NO_WORDS_LTE': UTTERANCE_LEN_IN_NO_WORDS_LTE,
    'UTTERANCE_LEN_IN_NO_PHRASES_EQ': UTTERANCE_LEN_IN_NO_PHRASES_EQ,
    'UTTERANCE_LEN_IN_NO_PHRASES_LTE': UTTERANCE_LEN_IN_NO_PHRASES_LTE,
}

SEGMENTAL_GROUPS = {
    'SEGMENTAL_POSITIONAL_ABSOLUTE': SEGMENTAL_POSITIONAL_ABSOLUTE,
    'SEGMENTAL_QUALITATIVE': SEGMENTAL_QUALITATIVE,
}

SYLLABIC_GROUPS = {
    'SYLLABIC_POSITIONAL_ABSOLUTE': SYLLABIC_POSITIONAL_ABSOLUTE,
    'SYLLABIC_POSITIONAL_RELATIVE': SYLLABIC_POSITIONAL_RELATIVE,
    'SYLLABIC_QUALITATIVE': SYLLABIC_QUALITATIVE,
    'SYLLABIC_COMPOSITIONAL': SYLLABIC_COMPOSITIONAL,
    'SYLLABIC_PARENTAL_COMPOSITION': SYLLABIC_PARENTAL_COMPOSITION,
}

WORD_GROUPS = {
    'WORD_POSITIONAL_ABSOLUTE': WORD_POSITIONAL_ABSOLUTE,
    'WORD_POSITIONAL_RELATIVE': WORD_POSITIONAL_RELATIVE,
    'WORD_COMPOSITIONAL': WORD_COMPOSITIONAL,
    'WORD_PARENTAL_COMPOSITION': WORD_PARENTAL_COMPOSITION,
}

PHRASAL_GROUPS = {
    'PHRASAL_POSITIONAL_ABSOLUTE': PHRASAL_POSITIONAL_ABSOLUTE,
    'PHRASAL_COMPOSITIONAL': PHRASAL_COMPOSITIONAL,
}

POSITIONAL_ABSOLUTE_GROUPS = {
    'SEGMENTAL_POSITIONAL_ABSOLUTE': SEGMENTAL_POSITIONAL_ABSOLUTE,
    'SYLLABIC_POSITIONAL_ABSOLUTE': SYLLABIC_POSITIONAL_ABSOLUTE,
    'WORD_POSITIONAL_ABSOLUTE': WORD_POSITIONAL_ABSOLUTE,
    'PHRASAL_POSITIONAL_ABSOLUTE': PHRASAL_POSITIONAL_ABSOLUTE,
}

POSITIONAL_RELATIVE_GROUPS = {
    'SYLLABIC_POSITIONAL_RELATIVE': SYLLABIC_POSITIONAL_RELATIVE,
    'WORD_POSITIONAL_RELATIVE': WORD_POSITIONAL_RELATIVE,
}

QUALITATIVE_GROUPS = {
    'SEGMENTAL_QUALITATIVE': SEGMENTAL_QUALITATIVE,
    'SYLLABIC_QUALITATIVE': SYLLABIC_QUALITATIVE,
}

COMPOSITIONAL_GROUPS = {
    'SYLLABIC_COMPOSITIONAL': SYLLABIC_COMPOSITIONAL,
    'WORD_COMPOSITIONAL': WORD_COMPOSITIONAL,
    'PHRASAL_COMPOSITIONAL': PHRASAL_COMPOSITIONAL,
    'UTTERANCE_COMPOSITIONAL': UTTERANCE_COMPOSITIONAL,
}

PARENTAL_COMPOSITION_GROUPS = {
    'SYLLABIC_PARENTAL_COMPOSITION': SYLLABIC_PARENTAL_COMPOSITION,
    'WORD_PARENTAL_COMPOSITION': WORD_PARENTAL_COMPOSITION,
}

LINGUISTIC_LEVEL_GROUPS = {
    'SEGMENTAL': SEGMENTAL,
    'SYLLABIC': SYLLABIC,
    'WORD': WORD,
    'PHRASAL': PHRASAL,
    'UTTERANCE': UTTERANCE
}

FEATURE_TYPE_GROUPS = {
    'POSITIONAL_ABSOLUTE': POSITIONAL_ABSOLUTE,
    'POSITIONAL_RELATIVE': POSITIONAL_RELATIVE,
    'QUALITATIVE': QUALITATIVE,
    'COMPOSITIONAL': COMPOSITIONAL,
    'PARENTAL_COMPOSITION': PARENTAL_COMPOSITION,
}

LINGUISTIC_LEVEL_WITH_FEATURE_TYPE_GROUPS = {
    'SEGMENTAL_POSITIONAL_ABSOLUTE': SEGMENTAL_POSITIONAL_ABSOLUTE,
    'SYLLABIC_POSITIONAL_ABSOLUTE': SYLLABIC_POSITIONAL_ABSOLUTE,
    'WORD_POSITIONAL_ABSOLUTE': WORD_POSITIONAL_ABSOLUTE,
    'PHRASAL_POSITIONAL_ABSOLUTE': PHRASAL_POSITIONAL_ABSOLUTE,
    'SYLLABIC_POSITIONAL_RELATIVE': SYLLABIC_POSITIONAL_RELATIVE,
    'WORD_POSITIONAL_RELATIVE': WORD_POSITIONAL_RELATIVE,
    'SEGMENTAL_QUALITATIVE': SEGMENTAL_QUALITATIVE,
    'SYLLABIC_QUALITATIVE': SYLLABIC_QUALITATIVE,
    'SYLLABIC_COMPOSITIONAL': SYLLABIC_COMPOSITIONAL,
    'WORD_COMPOSITIONAL': WORD_COMPOSITIONAL,
    'PHRASAL_COMPOSITIONAL': PHRASAL_COMPOSITIONAL,
    'UTTERANCE_COMPOSITIONAL': UTTERANCE_COMPOSITIONAL,
    'SYLLABIC_PARENTAL_COMPOSITION': SYLLABIC_PARENTAL_COMPOSITION,
    'WORD_PARENTAL_COMPOSITION': WORD_PARENTAL_COMPOSITION,
}