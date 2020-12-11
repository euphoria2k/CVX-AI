import Vue from 'vue';
export declare class AnnotationsDirective extends Vue {
    render(): void;
}
export declare const AnnotationsPlugin: {
    name: string;
    install(Vue: any): void;
};
/**
 * `e-annotation` directive represent a annotation of the VueJS Chart.
 * It must be contained in a Chart component(`ejs-chart`).
 * ```vue
 * <ejs-chart>
 *   <e-annotations>
 *    <e-annotation content='ID' />
 *    <e-annotation content='ID' />
 *   </e-annotations>
 * </ejs-chart>
 * ```
 */
export declare class AnnotationDirective extends Vue {
    render(): void;
}
export declare const AnnotationPlugin: {
    name: string;
    install(Vue: any): void;
};
